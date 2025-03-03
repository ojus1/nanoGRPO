import torch
from transformers import AutoTokenizer
from qwen2_latent import Qwen2ForCausalLM
from qwen2_latent_config import Qwen2Config

# Initialize model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Using smaller model for testing
config = Qwen2Config.from_pretrained(model_name)
model = Qwen2ForCausalLM.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Original length of tokenizer: ", len(tokenizer))
print("Embedding length of model: ", model.get_input_embeddings().weight.shape[0])

special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
num_added = tokenizer.add_special_tokens(special_tokens)

# Get the IDs of the newly added tokens
model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")
print("Added <bot> token id: ", model.bot_token_id)
print("Added <eot> token id: ", model.eot_token_id)
print("eos_token_id: ", tokenizer.eos_token_id)
print("bos_token_id: ", tokenizer.bos_token_id)

print("New length of tokenizer: ", len(tokenizer))
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


def test_latent_reasoning():

    def run_test(prompt, num_thoughts=3, expected_mode="latent"):
        print(f"\nTesting with prompt: {prompt}")
        print(f"Number of thoughts: {num_thoughts}")
        
        # Prepare input with latent reasoning tags
        if expected_mode == "latent":
            input_text = f"{prompt} <bot><num_thoughts={num_thoughts}>"
        else:
            input_text = prompt
            
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate with latent reasoning
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                input_text=input_text,
                return_latent_states=True if expected_mode == "latent" else False,
            )
            
            # Verify the shapes and masks only for latent mode
            if expected_mode == "latent":
                print("Latent mode outputs:", outputs.keys())  # Debug print
                input_embeds = outputs['input_embeds']
                latent_mask = outputs['latent_mask']
                print(f"Full sequence embeddings shape: {input_embeds.shape}")
                print(f"Latent mask shape: {latent_mask.shape}")
                print(f"Number of latent tokens: {latent_mask.sum().item()}")
                assert latent_mask.sum().item() == num_thoughts, "Incorrect number of latent tokens"
                decoded = tokenizer.decode(outputs["answer_token_ids"][0], skip_special_tokens=True)
            else:
                print("Normal mode output shape:", outputs.shape)  # Debug print
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"Generated text: {decoded}")
            
            return outputs

    # Test 1: Basic latent reasoning
    prompt = "Solve this math problem: 12 + 15 = ? Say 'Hi' before anything else."
    outputs_latent = run_test(prompt, num_thoughts=3, expected_mode="latent")
    
    # Test 2: Normal generation mode
    outputs_normal = run_test(prompt, expected_mode="normal")
    
    # Test 3: Longer latent sequence
    prompt = "Explain quantum entanglement:"
    outputs_long = run_test(prompt, num_thoughts=5, expected_mode="latent")

    print("\nAll tests completed successfully!")

def test_latent_probing():
    # Test probing latent states
    prompt = "Explain quantum entanglement: <bot><num_thoughts=3>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        # Generate with probing
        outputs = model.generate_with_probe(
            inputs.input_ids,
            input_text=prompt,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Verify output structure
        assert isinstance(outputs, dict)
        assert 'generation_output' in outputs
        assert 'probed_logits' in outputs
        assert 'probed_tokens' in outputs
        
        # Check shapes
        probed_logits = outputs['probed_logits']
        probed_tokens = outputs['probed_tokens']
        assert probed_logits.dim() == 3  # (batch, num_latents, vocab_size)
        assert probed_tokens.dim() == 2  # (batch, num_latents)
        
        # Decode probed tokens
        decoded_probes = model._decode_probed_tokens(probed_tokens)
        print("\nProbed latent states decoded as:")
        for i, text in enumerate(decoded_probes):
            print(f"Latent {i}: token_id: {probed_tokens[0][i]}, text: {text}")
        
        # Verify normal generation still works
        answer = tokenizer.decode(outputs['generation_output']['answer_token_ids'][0])
        print(f"\nFinal answer: {answer}")
        
def test_latent_gradients():
    # Set model to training mode
    model.train()
    
    # Prepare input with latent reasoning
    prompt = "Solve this math problem: 12 + 15 = ? <bot><num_thoughts=3>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Get input embeddings
    input_embeds = model.get_input_embeddings()(inputs.input_ids.to(device))
    print("input_embeds.requires_grad before forward:", input_embeds.requires_grad)
    
    # Forward pass with gradient tracking
    outputs = model(
        inputs_embeds=input_embeds,
        input_text=prompt,
        output_hidden_states=True,
        # **kwargs
    )
    
    # Get hidden states from the output
    hidden_states = outputs.hidden_states[-1]  # Get last layer's hidden states
    hidden_states.retain_grad()
    print("hidden_states.requires_grad after forward:", hidden_states.requires_grad)
    logits = outputs.logits
    
    # Generate a "target" sequence (this would normally be your desired output)
    target_text = "The answer is 27."
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(device)
    
    # Calculate a simple reward (could be based on any metric)
    # Here we'll use a dummy reward of 1.0
    reward = torch.tensor(1.0).to(device)
    
    # Calculate REINFORCE-like loss
    # Log prob of generated sequence
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    selected_log_probs = torch.gather(
        log_probs, 
        -1, 
        target_ids[:, :log_probs.size(1)].unsqueeze(-1)
    ).squeeze(-1)
    
    # Only consider loss for tokens after the prompt
    sequence_loss = -(selected_log_probs * reward).mean()
    
    # Backpropagate
    sequence_loss.backward()
    
    # Check gradients for hidden states
    hidden_grads = hidden_states.grad
    if hidden_grads is not None:
        grad_magnitudes = hidden_grads.abs().mean(dim=-1)  # Average over hidden dimension
        print("\nGradient magnitudes across sequence:")
        print(grad_magnitudes)
        
        # Verify gradients exist and are non-zero
        assert hidden_grads.abs().sum() > 0, "No gradients found in hidden states"
        print("\nGradient check passed successfully!")
    else:
        print("\nWarning: No gradients found in hidden states")
    
    # Reset model to eval mode
    model.eval()

def test_latent_noise_injection():
    """Test the noise injection functionality for latent reasoning"""
    print("\nTesting noise injection for latent reasoning...")
    
    # Prepare a test prompt for latent generation
    prompt = "What is the capital of France? <bot><num_thoughts=3>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Set non-zero values in the Explorer for testing purposes
    with torch.no_grad():
        # Initialize with small but non-zero values to ensure visible differences
        model.model.explorer.mean.fill_(0.01)
        diag_indices = (model.model.explorer.tril_indices[0] == model.model.explorer.tril_indices[1])
        model.model.explorer.cholesky_factors[diag_indices] = 0.1
    
    # Generate with no noise first
    with torch.inference_mode():
        baseline_outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            eos_token_id=tokenizer.eos_token_id,
            input_text=prompt,
            return_latent_states=True,
            noise_scale=0.0,  # No noise
        )
        
        # Now generate with noise
        noisy_outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            eos_token_id=tokenizer.eos_token_id,
            input_text=prompt,
            return_latent_states=True,
            noise_scale=5.0,  # Substantial noise
        )
    
    # Extract latent states from both outputs
    baseline_latents = baseline_outputs['latent_states']
    noisy_latents = noisy_outputs['latent_states']
    
    # Verify outputs have the expected structure
    assert 'latent_states' in baseline_outputs
    assert 'latent_states' in noisy_outputs
    assert 'answer_token_ids' in baseline_outputs
    assert 'answer_token_ids' in noisy_outputs
    
    # Compare latent states - they should be different with noise
    latent_diff = (baseline_latents - noisy_latents).abs().mean().item()
    print(f"Average difference between baseline and noisy latents: {latent_diff:.6f}")
    assert latent_diff > 0, "Noise injection had no effect on latent states"
    
    # Decode the answer tokens to compare outputs
    baseline_text = tokenizer.decode(baseline_outputs['answer_token_ids'][0], skip_special_tokens=True)
    noisy_text = tokenizer.decode(noisy_outputs['answer_token_ids'][0], skip_special_tokens=True)
    print(f"Baseline output: {baseline_text}")
    print(f"Noisy output: {noisy_text}")
    
    # Test with gradient tracking (training mode)
    model.train()
    
    # Create embeddings that require gradients
    test_embed = model.get_input_embeddings()(inputs.input_ids.to(device))
    test_embed.requires_grad_(True)
    
    # Create a dummy latent token
    dummy_latent = torch.randn(1, 1, model.config.hidden_size, device=device)
    dummy_latent.requires_grad_(True)
    
    # Test noise injection with gradients
    noisy_latent = model._inject_noise_to_latent(dummy_latent, noise_scale=0.1)
    assert noisy_latent.requires_grad, "Noisy latent lost gradient information"
    
    # Compute a dummy loss and check if gradients flow
    dummy_loss = noisy_latent.sum()
    dummy_loss.backward()
    
    assert dummy_latent.grad is not None, "Gradients did not flow through the noise injection"
    print("Gradient check passed: noise injection is differentiable")
    
    # Reset model to eval mode
    model.eval()
    
    print("Noise injection test completed successfully!")

def test_batch_latent_reasoning():
    print("\nTesting batch latent generation...")

    # Prepare a batch of latent prompts with consistent latent reasoning tags
    prompts_latent = [
        "What is the capital of France? <bot><num_thoughts=3>",
        "Solve 5 + 7 = ? <bot><num_thoughts=3>",
        "Who wrote Romeo and Juliet? <bot><num_thoughts=3>"
    ]
    
    # Validate latent configuration extraction for one sample
    try:
        latent_config = model._extract_latent_config(prompts_latent[0])
        assert latent_config is not None, "Failed to extract latent config from valid input"
        print("Validated consistent latent prompts successfully")
    except ValueError:
        assert False, "Should not raise ValueError for consistent latent prompts"

    # Prepare a batch with consistent normal reasoning (no latent tags)
    prompts_normal = [
        "What is the capital of France?",
        "Solve 5 + 7 = ?",
        "Who wrote Romeo and Juliet?"
    ]
    try:
        latent_config = model._extract_latent_config(prompts_normal[0])
        assert latent_config is None, "Should not extract latent config from normal input"
        print("Validated consistent normal prompts successfully")
    except ValueError:
        assert False, "Should not raise ValueError for consistent normal prompts"
    
    # --- Batch Generation for latent mode ---
    inputs_latent = tokenizer(prompts_latent, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        batch_outputs_latent = model.generate(
            inputs_latent.input_ids,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            input_text=prompts_latent,  # Passing list of prompts for reference if needed
            return_latent_states=True,
        )
    
    # Check each output in the latent batch
    for i in range(len(prompts_latent)):
        latent_mask = batch_outputs_latent['latent_mask'][i]
        latent_count = latent_mask.sum().item()
        assert latent_count == 3, f"Sample {i}: Expected 3 latent tokens, got {latent_count}"
        answer_decoded = tokenizer.decode(batch_outputs_latent['answer_token_ids'][i], skip_special_tokens=True)
        print(f"Latent sample {i} generated output: {answer_decoded}")
    
    # --- Batch Generation for normal mode ---
    inputs_normal = tokenizer(prompts_normal, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        batch_outputs_normal = model.generate(
            inputs_normal.input_ids,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            input_text=prompts_normal,  # Passing list of prompts for reference if needed
            return_latent_states=False,
        )
    
    # Check and print outputs for normal batch generation
    # If batch_outputs_normal is a tensor of shape (batch, tokens), iterate through the first dimension
    for i in range(len(prompts_normal)):
        answer_decoded = tokenizer.decode(batch_outputs_normal[i], skip_special_tokens=True)
        print(f"Normal sample {i} generated output: {answer_decoded}")
    
    print("Batch latent generation test completed successfully!")

def test_mixed_batch_error():
    print("\nTesting mixed batch handling...")
    
    # Prepare a batch with mixed modes
    prompts = [
        "What is the capital of France? <bot><num_thoughts=3>",  # Latent mode
        "Solve 5 + 7 = ?",  # Normal mode (no latent tag)
        "Who wrote Romeo and Juliet? <bot><num_thoughts=3>"  # Latent mode
    ]
    
    # Test our validation logic directly
    try:
        # Check if any prompt has latent tag
        has_latent = ["<bot><num_thoughts=" in txt for txt in prompts]
        if not all(has_latent) and not all(not x for x in has_latent):
            raise ValueError("Mixed generation modes in batch detected")
        assert False, "Mixed batch should have raised an error but didn't!"
    except ValueError as e:
        print(f"Caught expected error: {e}")
        assert "Mixed generation modes" in str(e), f"Unexpected error message: {e}"
    
    print("Mixed batch error test passed!")

def test_forced_token_generation():
    print("\nTesting forced token generation...")
    
    # Test 1: Forced generation in normal mode
    normal_prompt = "What is the capital of France?"
    inputs_normal = tokenizer(normal_prompt, return_tensors="pt").to(device)
    
    # Create forced tokens - we'll force "Paris" as the answer
    forced_tokens = tokenizer(" Paris", return_tensors="pt").input_ids.to(device)
    forced_token_list = [t.squeeze(0) for t in forced_tokens.split(1, dim=1)]
    
    # Generate with forced tokens
    with torch.inference_mode():
        normal_outputs = model.generate(
            inputs_normal.input_ids,
            max_new_tokens=20,
            eos_token_id=tokenizer.eos_token_id,
            input_text=normal_prompt,
            forced_token_ids=forced_token_list,
        )
    
    # Decode and verify output contains our forced tokens
    normal_text = tokenizer.decode(normal_outputs[0], skip_special_tokens=True)
    print(f"Normal generation with forced tokens: {normal_text}")
    forced_text = tokenizer.decode(forced_tokens[0], skip_special_tokens=True)
    assert forced_text in normal_text, f"Forced text '{forced_text}' not found in output '{normal_text}'"
    
    # Test 2: Forced generation in latent mode
    latent_prompt = "What is the capital of France? <bot><num_thoughts=3>"
    inputs_latent = tokenizer(latent_prompt, return_tensors="pt").to(device)
    
    # Generate with forced tokens in latent mode
    with torch.inference_mode():
        latent_outputs = model.generate(
            inputs_latent.input_ids,
            max_new_tokens=20,
            eos_token_id=tokenizer.eos_token_id,
            input_text=latent_prompt,
            forced_token_ids=forced_token_list,
            return_latent_states=True,
        )
    
    # Decode and verify output contains our forced tokens
    latent_text = tokenizer.decode(latent_outputs['answer_token_ids'][0], skip_special_tokens=True)
    print(f"Latent generation with forced tokens: {latent_text}")
    assert forced_text in latent_text, f"Forced text '{forced_text}' not found in latent output '{latent_text}'"
    
    # Test 3: Batch forced generation
    batch_prompts = [
        "What is the capital of France?",
        "What is the capital of Italy?",
        "What is the capital of Spain?"
    ]
    inputs_batch = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
    
    # Create forced tokens for batch - we'll force "The capital" for all examples
    forced_batch_tokens = tokenizer(" The capital", return_tensors="pt").input_ids.to(device)
    # Expand to batch size
    forced_batch_tokens = forced_batch_tokens.expand(len(batch_prompts), -1)
    forced_batch_list = [t for t in forced_batch_tokens.unbind(dim=1)]
    
    # Generate with forced tokens in batch mode
    with torch.inference_mode():
        batch_outputs = model.generate(
            inputs_batch.input_ids,
            max_new_tokens=20,
            eos_token_id=tokenizer.eos_token_id,
            input_text=batch_prompts,
            forced_token_ids=forced_batch_list,
        )
    
    # Verify all outputs contain forced text
    batch_forced_text = tokenizer.decode(forced_batch_tokens[0], skip_special_tokens=True)
    for i, output in enumerate(batch_outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Batch item {i} with forced tokens: {text}")
        assert batch_forced_text in text, f"Forced text not found in batch item {i}"
    
    # Test 4: Mixed-length forced tokens (fixed)
    shorter_forced = tokenizer(" The", return_tensors="pt").input_ids.to(device).squeeze(0)
    longer_forced = tokenizer(" capital of", return_tensors="pt").input_ids.to(device).squeeze(0)
    mixed_forced_tokens = torch.cat([shorter_forced, longer_forced], dim=0)
    mixed_forced_list = [t.unsqueeze(0) for t in mixed_forced_tokens]

    # Generate with mixed-length forced tokens
    with torch.inference_mode():
        mixed_outputs = model.generate(
            inputs_normal.input_ids,  # Using first prompt
            max_new_tokens=20,
            eos_token_id=tokenizer.eos_token_id,
            input_text=normal_prompt,
            forced_token_ids=mixed_forced_list,
        )

    mixed_text = tokenizer.decode(mixed_outputs[0], skip_special_tokens=True)
    mixed_forced_text = tokenizer.decode(mixed_forced_tokens, skip_special_tokens=True)
    print(f"Mixed-length forced tokens: {mixed_text}")
    assert mixed_forced_text in mixed_text, f"Mixed forced text '{mixed_forced_text}' not found in output '{mixed_text}'"
    
    print("Forced token generation tests completed successfully!")

def test_explorer_gradients():
    """Test if Explorer parameters receive gradients during training"""
    print("\nTesting Explorer gradient flow...")
    
    # Set model to training mode
    model.train()
    
    # Create a dummy latent token directly 
    batch_size = 1
    hidden_size = model.config.hidden_size
    dummy_latent = torch.zeros(batch_size, 1, hidden_size, device=device, requires_grad=True)
    
    # Access the explorer directly
    model.model.explorer.mean.retain_grad()
    model.model.explorer.cholesky_factors.retain_grad()
    
    # Apply noise to latent tokens - direct call to avoid any no_grad blocks
    noisy_latent = model.model.explorer.sample(dummy_latent, scale=1.0, deterministic=False)
    
    # Create a dummy loss (different from zero to ensure gradient flow)
    loss = noisy_latent.pow(2).mean()
    
    # Backpropagate
    loss.backward()
    
    # Check if Explorer parameters received gradients
    if model.model.explorer.mean.grad is not None:
        print("Explorer mean grad norm:", model.model.explorer.mean.grad.norm().item())
        assert model.model.explorer.mean.grad.norm() > 0, "Explorer mean gradients are zero"
    else:
        assert False, "Explorer mean didn't receive gradients"
        
    if model.model.explorer.cholesky_factors.grad is not None:
        print("Explorer cholesky factors grad norm:", model.model.explorer.cholesky_factors.grad.norm().item())
        assert model.model.explorer.cholesky_factors.grad.norm() > 0, "Explorer cholesky factors gradients are zero"
    else:
        assert False, "Explorer cholesky factors didn't receive gradients"
    
    print("Explorer gradient test passed successfully!")
    
    # Reset model to eval mode
    model.eval()

def test_explorer_deterministic_mode():
    """Test if Explorer produces deterministic outputs in eval mode"""
    print("\nTesting Explorer deterministic behavior...")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Prepare test input
    prompt = "What is the capital of France? <bot><num_thoughts=3>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Create a dummy latent token
    dummy_latent = torch.zeros(1, 1, model.config.hidden_size, device=device)
    
    # Generate latent tokens with Explorer twice
    with torch.inference_mode():
        noisy_latent1 = model._inject_noise_to_latent(dummy_latent, noise_scale=1.0)
        noisy_latent2 = model._inject_noise_to_latent(dummy_latent, noise_scale=1.0)
    
    # In eval mode with deterministic=True, outputs should be identical
    diff = (noisy_latent1 - noisy_latent2).abs().max().item()
    print(f"Maximum difference between two deterministic samples: {diff:.10f}")
    assert diff < 1e-5, "Explorer is not deterministic in eval mode"
    
    # Switch to training mode where it should be stochastic
    model.train()
    noisy_latent1 = model._inject_noise_to_latent(dummy_latent, noise_scale=1.0)
    noisy_latent2 = model._inject_noise_to_latent(dummy_latent, noise_scale=1.0)
    
    # In training mode, outputs should differ (stochastic)
    diff = (noisy_latent1 - noisy_latent2).abs().max().item()
    print(f"Maximum difference between two stochastic samples: {diff:.10f}")
    assert diff > 1e-5, "Explorer is not stochastic in training mode"
    
    # Reset to eval mode
    model.eval()
    
    print("Explorer deterministic behavior test passed!")

if __name__ == "__main__":
    test_latent_reasoning()
    test_latent_probing()
    test_latent_gradients()
    test_latent_noise_injection()
    test_batch_latent_reasoning()
    test_mixed_batch_error()
    test_forced_token_generation()
    test_explorer_gradients()
    test_explorer_deterministic_mode() 