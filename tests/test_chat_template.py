import sys
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path to import elitelm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elitelm import ChatSession, AppConfig

def test_chat_templates():
    print("Testing Chat Templates...")
    
    # Mock config
    config = AppConfig(model="dummy_model")
    
    # Mock tokenizer and model loading
    with patch("elitelm._load_model") as mock_load:
        mock_tokenizer = MagicMock()
        # Remove apply_chat_template to force custom template usage
        del mock_tokenizer.apply_chat_template 
        mock_load.return_value = (MagicMock(), mock_tokenizer, MagicMock())
        
        # 1. Test Llama 3 Template
        print("\n[Test] Llama 3 Template")
        config.chat_template = "llama3"
        session = ChatSession(config)
        session.chat_history = [{"role": "user", "content": "Hello"}]
        prompt, _ = session._build_prompt("How are you?")
        
        expected_llama3 = (
            "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nHow are you?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        if prompt == expected_llama3:
            print("✅ Llama 3 template matches")
        else:
            print(f"❌ Llama 3 template mismatch:\nGot: {repr(prompt)}\nExp: {repr(expected_llama3)}")

        # 2. Test Phi-3 Template
        print("\n[Test] Phi-3 Template")
        config.chat_template = "phi3"
        session = ChatSession(config) # Re-init to pick up config change if needed, though we just changed the obj
        session.chat_history = [{"role": "user", "content": "Hello"}]
        prompt, _ = session._build_prompt("How are you?")
        
        expected_phi3 = (
            "<user>\nHello<|end|>\n"
            "<user>\nHow are you?<|end|>\n"
            "<|assistant|>\n"
        )
        
        if prompt == expected_phi3:
            print("✅ Phi-3 template matches")
        else:
            print(f"❌ Phi-3 template mismatch:\nGot: {repr(prompt)}\nExp: {repr(expected_phi3)}")

        # 3. Test ChatML Template
        print("\n[Test] ChatML Template")
        config.chat_template = "chatml"
        session = ChatSession(config)
        session.chat_history = [{"role": "user", "content": "Hello"}]
        prompt, _ = session._build_prompt("How are you?")
        
        expected_chatml = (
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>user\nHow are you?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        if prompt == expected_chatml:
            print("✅ ChatML template matches")
        else:
            print(f"❌ ChatML template mismatch:\nGot: {repr(prompt)}\nExp: {repr(expected_chatml)}")

if __name__ == "__main__":
    test_chat_templates()
