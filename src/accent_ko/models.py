"""Adapters based on author code and official model interfaces, loaded lazily.

Weight downloads/inference only happen when run/evaluate is explicitly invoked.
"""
import copy
import re


def require_revision(spec):
    if not re.fullmatch(r'[0-9a-f]{40}', spec.get('revision') or ''):
        raise ValueError(f"Pin a 40-character Hugging Face commit for {spec['repo']}")


def messages(text, audio, system, kind):
    if kind == 'qwen2':
        content = [{'type': 'audio', 'audio_url': 'local.wav'}] if audio is not None else [{'type': 'text', 'text': text}]
    elif kind == 'meralion':
        content = '<SpeechHere>' if audio is not None else text
    elif kind == 'minicpm':
        content = [audio] if audio is not None else [text]
    else:
        content = text
    return ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': content}]


class AudioModel:
    def __init__(self, name, spec, device='cuda'):
        require_revision(spec)
        import torch
        import transformers as hf
        self.torch, self.name, self.spec, self.device = torch, name, spec, device
        args = {'revision': spec['revision'], 'trust_remote_code': True}
        repo = spec['repo']
        if name == 'qwen2':
            self.processor = hf.AutoProcessor.from_pretrained(repo, **args)
            self.model = hf.Qwen2AudioForConditionalGeneration.from_pretrained(repo, torch_dtype='auto', **args).to(device).eval()
        elif name == 'meralion':
            self.processor = hf.AutoProcessor.from_pretrained(repo, **args)
            self.model = hf.AutoModelForSpeechSeq2Seq.from_pretrained(repo, torch_dtype=torch.bfloat16, **args).to(device).eval()
        elif name == 'minicpm':
            if not spec.get('assistant_prompt_ko'):
                raise ValueError('Set MiniCPM assistant_prompt_ko to the translated audio_assistant prompt')
            self.processor = hf.AutoTokenizer.from_pretrained(repo, **args)
            self.model = hf.AutoModel.from_pretrained(repo, torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa', init_vision=True, init_audio=True, init_tts=False,
                    **args).to(device).eval()
        elif name == 'diva':
            # DiVA's custom loader ignores revision for its own weights. Load a pinned local snapshot.
            from huggingface_hub import snapshot_download
            local = snapshot_download(repo, revision=spec['revision'])
            self.model = hf.AutoModel.from_pretrained(local, trust_remote_code=True,
                    speech_encoder_device=device, device_map=device).eval()
        elif name == 'ultravox':
            self.pipe = hf.pipeline(model=repo, device=device, **args)
        else:
            raise ValueError(f'Unknown model {name}')

    def generate(self, text, audio=None, system=None):
        torch, name = self.torch, self.name
        gen = copy.deepcopy(self.spec.get('text_generation', self.spec['generation'])
                            if audio is None else self.spec['generation'])
        with torch.inference_mode():
            if name == 'diva':
                if system:
                    raise ValueError('DiVA is excluded from the paper defense experiment')
                if audio is not None:
                    return self.model.generate([audio], **gen)[0]
                m = self.model
                # Match the author's text prefix construction with native decoder.generate.
                tokens = m.tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(m.pre_system.device)
                prefix = torch.cat([m.pre_system, tokens, m.post_system, m.final_header], dim=1)
                embedding = m.llm_decoder.model.embed_tokens(prefix)
                result = m.llm_decoder.generate(inputs_embeds=embedding, do_sample=False,
                        eos_token_id=128009, **gen)
                return m.tokenizer.decode(result[0], skip_special_tokens=True)
            if name == 'ultravox':
                turns = [{'role': 'system', 'content': system or self.spec['system_ko']}]
                payload = {'turns': turns}
                if audio is None:
                    turns.append({'role': 'user', 'content': text})
                else:
                    payload.update(audio=audio, sampling_rate=16000)
                result = self.pipe(payload, **gen)
                if not isinstance(result, str):
                    raise ValueError('Unexpected Ultravox output schema; refuse to stringify metadata')
                return result
            if name == 'minicpm':
                chat = messages(text, audio, system, name)
                offset = 1 if system else 0
                chat.insert(offset, {'role': 'user', 'content': self.spec['assistant_prompt_ko']})
                return self.model.chat(msgs=chat, tokenizer=self.processor,
                        use_tts_template=True, generate_audio=False, **gen)
            chat = messages(text, audio, system, name)
            tokenizer = self.processor if name == 'qwen2' else self.processor.tokenizer
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if name == 'qwen2':
                inputs = self.processor(text=prompt, audios=[audio] if audio is not None else None,
                                        return_tensors='pt', padding=True)
            elif audio is None:
                inputs = self.processor.tokenizer(prompt, return_tensors='pt')
            else:
                if len(audio) > 16000 * 30:
                    raise ValueError('MERaLiON input exceeds documented 30 seconds; no silent truncation')
                inputs = self.processor(text=prompt, audios=audio)
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if name == 'meralion' and 'input_features' in inputs:
                inputs['input_features'] = inputs['input_features'].to(self.model.dtype)
            result = self.model.generate(**inputs, **gen)
            result = result[:, inputs['input_ids'].shape[1]:]
            return self.processor.batch_decode(result, skip_special_tokens=True)[0]


class TextJudge:
    def __init__(self, spec, device='cuda'):
        require_revision(spec)
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        args = {'revision': spec['revision']}
        self.tokenizer = AutoTokenizer.from_pretrained(spec['repo'], **args)
        self.model = AutoModelForCausalLM.from_pretrained(spec['repo'], torch_dtype='auto', **args).to(device).eval()
        self.torch, self.device, self.spec = torch, device, spec

    def generate(self, chat):
        tokens = self.tokenizer.apply_chat_template(
            chat, return_tensors='pt',
            add_generation_prompt='Llama-Guard' not in self.spec['repo']).to(self.device)
        with self.torch.inference_mode():
            result = self.model.generate(input_ids=tokens, **self.spec['generation'])
        return self.tokenizer.decode(result[0][tokens.shape[-1]:], skip_special_tokens=True)
