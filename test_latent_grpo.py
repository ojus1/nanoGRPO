import random
import torch
from transformers import AutoTokenizer
from qwen2_latent import Qwen2ForCausalLM
from datasets import Dataset
from coconut_grpo import GRPO
import time
from peft import get_peft_model, LoraConfig

def random_reward(sample, response):
    # Simple reward function that checks for correct answers in the response
    if "2+2=4" in response or "3+3=6" in response:
        return 1.0
    else:
        return 0.0

def test_latent_grpo_components():
    """Tests core components of the GRPO latent reasoning implementation"""
    # Initialize model and tokenizer (using a small version for testing)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = Qwen2ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens for latent reasoning
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
    model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")

        
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'v_proj'],
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)

    # Setup a minimal dataset for testing
    test_data = [
        {"prompt": [{"role": "user", "content": "What is 2+2?"}]},
        {"prompt": [{"role": "user", "content": "What is 3+3?"}]}
    ]
    test_data = test_data * 100  # Extend to have enough data
    dataset = Dataset.from_list(test_data)

    # Use a simple reward function
    def random_reward(sample, response):
        if "2+2=4" in response or "3+3=6" in response:
            return 1.0
        else:
            return 0.0

    reward_funcs = [random_reward]

    # Instantiate GRPO with latent parameters
    trainer = GRPO(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        group_size=4,
        batch_size=2,
        max_iterations=10,
        dataset=dataset,
        reward_functions=reward_funcs,
        log_wandb=False,
        beta=0.0,
        epsilon=0.1,
        max_continuous_tokens=2,  # Use a smaller value for testing
    )

    # Test the _sample_continuous_tokens method
    continuous_tokens = trainer._sample_continuous_tokens()
    print(f"Continuous tokens sampled: {continuous_tokens}")
    assert 1 <= continuous_tokens <= trainer.max_continuous_tokens, "Continuous tokens out of range"
    
    # Test the sample_batch method which is the core of the implementation
    try:
        batch_result = trainer.sample_batch()
        assert isinstance(batch_result, dict), "batch_result should be a dictionary"
        assert 'rewards' in batch_result, "batch_result should contain 'rewards'"
        assert 'loss_mask' in batch_result, "batch_result should contain 'loss_mask'"
        
        # Check if it has either input_ids (token mode) or inputs_embeds (latent mode)
        assert 'input_ids' in batch_result or 'inputs_embeds' in batch_result, \
            "batch_result should contain either 'input_ids' or 'inputs_embeds'"
        
        # For latent mode, verify the structure
        if 'inputs_embeds' in batch_result:
            assert batch_result['mode'] == 'latent', "Mode should be 'latent'"
            assert 'attention_mask' in batch_result, "Missing attention_mask in latent mode"
            assert 'answer_token_ids' in batch_result, "Missing answer_token_ids in latent mode"
            
            # Check shapes
            batch_size_total = trainer.batch_size * trainer.group_size
            assert batch_result['inputs_embeds'].shape[0] == batch_size_total, "Wrong batch dimension"
        
        print("✓ sample_batch runs successfully")
    except Exception as e:
        print(f"✗ sample_batch failed: {e}")

    # Test the compute_loss function with proper input
    try:
        if 'inputs_embeds' in batch_result:  # Latent mode
            # Extract a small subset for testing loss computation
            mini_batch = {
                'inputs_embeds': batch_result['inputs_embeds'][:2],
                'attention_mask': batch_result['attention_mask'][:2],
                'answer_token_ids': batch_result['answer_token_ids'][:2],
                'mode': 'latent'
            }
            
            dummy_rewards = torch.ones(2, device=mini_batch['inputs_embeds'].device)
            dummy_mean = torch.tensor(0.5, device=mini_batch['inputs_embeds'].device)
            dummy_std = torch.tensor(0.5, device=mini_batch['inputs_embeds'].device)
            loss_mask = (mini_batch['answer_token_ids'] != tokenizer.pad_token_id).float()
            
            loss, loss1, loss2 = trainer.compute_loss(
                mini_batch, None, dummy_rewards, dummy_mean, dummy_std, loss_mask
            )
            
            assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
            assert loss.numel() == 1, f"Loss should be a scalar, got shape {loss.shape}"
            print(f"✓ Compute loss returned valid loss: {loss.item()}")
            
            # Test gradient flow
            loss.backward()
            grad_exists = False
            for param in trainer.model.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    grad_exists = True
                    break
            assert grad_exists, "No gradients flowing to model parameters"
            print("✓ Gradients flow correctly through latent GRPO pipeline")
            
        else:  # Token mode
            print("Skipping loss computation test - not in latent mode")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"✗ Loss computation failed: {e}")

    # Test a mini training loop (just a few steps to verify it works)
    try:
        trainer.max_iterations = 2  # Set to a small number for quick testing
        trainer.train(max_iterations=2)
        print("✓ Training loop runs successfully")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"✗ Training loop failed: {e}")

def test_latent_to_token_transition():
    """Test the transition from latent token generation to answer token generation"""
    print("\nTesting latent-to-token transition...")
    
    # Initialize model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = Qwen2ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens for latent reasoning
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
    model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create batch inputs with different sequence lengths to stress test dimension handling
    prompts = [
        "Solve this math problem: 2+2=? <bot><num_thoughts=3>",
        "What is the capital of France? <bot><num_thoughts=3>",
    ]
    
    # Tokenize with padding to create uneven lengths
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    # Run generation with latent reasoning
    try:
        outputs = model.generate(
            inputs.input_ids,
            input_text=prompts,
            max_new_tokens=20,
            eos_token_id=tokenizer.eos_token_id,
            return_latent_states=True,
            noise_scale=0.5  # Add noise to test robustness
        )
        
        # Verify that we have all expected components
        assert 'input_embeds' in outputs
        assert 'latent_mask' in outputs
        assert 'answer_token_ids' in outputs
        
        # Check the shape of answer tokens
        answer_tokens = outputs['answer_token_ids']
        assert answer_tokens.shape[0] == len(prompts), "Batch dimension mismatch"
        
        # Decode and print results
        for i, tokens in enumerate(answer_tokens):
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"Input {i}: {prompts[i]}")
            print(f"Output {i}: {decoded}")
        
        print("✓ Latent to token transition test passed")
        
    except Exception as e:
        print(f"✗ Latent to token transition test failed: {e}")
        raise

def test_batch_latent_grpo():
    """Test batch processing with latent GRPO"""
    print("\nTesting batch processing with latent GRPO...")
    
    # Initialize model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = Qwen2ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens for latent reasoning
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
    model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")
    
    # Create a minimal dataset with VERY DIFFERENT prompt lengths to stress test
    test_data = [
        {"prompt": [{"role": "user", "content": "2+2?"}]},  # Very short
        {"prompt": [{"role": "user", "content": "What is the capital of France? This is a longer prompt with more tokens to process and analyze carefully for maximum difference in length."}]},  # Very long
    ] * 100
    dataset = Dataset.from_list(test_data)
    
    # Simple reward function
    def dummy_reward(sample, response):
        return 1.0 if len(response) > 5 else 0.0
    
    # Create GRPO trainer with batch size = 2 
    try:
        trainer = GRPO(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            group_size=4,
            batch_size=2,  # Test with batch_size > 1 and exactly match dataset size
            max_iterations=2,
            dataset=dataset,
            reward_functions=[dummy_reward],
            log_wandb=False,
            max_continuous_tokens=2,  # Use fewer tokens for faster test
        )
        
        # Run a sample batch explicitly
        sample_result = trainer.sample_batch()
        
        # Verify batch results
        assert 'rewards' in sample_result
        assert 'loss_mask' in sample_result
        
        # Check if we have latent-mode outputs
        if 'inputs_embeds' in sample_result:
            print("✓ Latent mode active in batch")
            # Verify embedding dimensions
            embeds = sample_result['inputs_embeds']
            mask = sample_result['loss_mask']
            assert embeds.dim() == 3, f"Expected 3D tensor for embeddings, got {embeds.dim()}"
            assert embeds.shape[0] == trainer.group_size * trainer.batch_size, "Batch dimension mismatch"
        
        # Now run a full training step (which does multiple sample_batch calls)
        trainer.train(max_iterations=1)
        print("✓ Batch latent GRPO test passed")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"✗ Batch latent GRPO test failed: {e}")
        raise

if __name__ == "__main__":
    start = time.time()
    torch.manual_seed(42)  # For reproducibility
    test_latent_grpo_components()
    test_latent_to_token_transition()
    test_batch_latent_grpo()
    # print("All latent GRPO tests passed! Time elapsed:", time.time() - start) 