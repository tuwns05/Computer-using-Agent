"""Regression: cached remote-model decoding must match full-sequence decoding."""
import unittest
from unittest.mock import Mock

import mage


class GenerationCompatibilityTest(unittest.TestCase):
    def test_notepad_completion_requires_actual_document(self):
        task = 'Mở Notepad và gõ Xin chào Mage'
        state = {'foreground_app': 'Notepad.exe', 'document_text': ['Xin chào Mage']}
        self.assertTrue(mage.verified_notepad_task(task, state))
        self.assertFalse(mage.verified_notepad_task(task, {**state, 'document_text': ['Xin eeeeeeeee']}))
        self.assertFalse(mage.verified_notepad_task(task, {**state, 'foreground_app': 'Code.exe'}))
        self.assertFalse(mage.verified_notepad_task(task, {'foreground_app': 'Notepad.exe', 'editor_text': ['Xin chào Mage']}))
        self.assertFalse(mage.verified_notepad_task('Mở Notepad và gõ Xin chào Mage rồi lưu file', state))

    def test_linux_launcher_uses_windows_run(self):
        desktop = mage.Desktop.__new__(mage.Desktop)
        desktop.user = Mock()
        desktop.keypress(['ALT', 'F2'])
        self.assertEqual([call.args[0] for call in desktop.user.keybd_event.call_args_list],
                         [91, ord('R'), ord('R'), 91])

    def test_cached_logits_match_full_sequence(self):
        import torch
        import transformers
        from transformers import AutoConfig, Qwen3Config
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        torch.manual_seed(42)
        path = mage.model_path()
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        config.text_config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=16)
        vision = config.vision_config
        vision.hidden_size = 128
        vision.out_hidden_size = vision.text_hidden_size = 64
        vision.intermediate_size = 128
        vision.num_hidden_layers = 1
        vision.num_attention_heads = 4
        config.image_token_id = 61
        config.video_token_id = 62
        config.vision_start_token_id = 60
        config.vision_end_token_id = 63
        cls = get_class_from_dynamic_module(config.auto_map['AutoModelForCausalLM'], path, local_files_only=True)
        model = cls(config).eval()
        mage.fix_generation_positions(model)
        tokens = torch.tensor([[5, 10, 15, 20]])
        with torch.inference_mode():
            full = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
            prefix = model(input_ids=tokens[:, :3], attention_mask=torch.ones_like(tokens[:, :3]), use_cache=True)
            generation_args = ({'next_sequence_length': 1, 'is_first_iteration': False}
                               if int(transformers.__version__.split('.')[0]) >= 5
                               else {'cache_position': torch.tensor([3])})
            inputs = model.prepare_inputs_for_generation(
                tokens, **generation_args, past_key_values=prefix.past_key_values,
                attention_mask=torch.ones_like(tokens), use_cache=True)
            cached = model(**inputs)
        self.assertEqual(cached.logits.shape[1], 1)
        torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], atol=1e-5, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
