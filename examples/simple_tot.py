from rl_tot.core.agent import ToTAgent


def main():
    # Initialize agent with GPT-2 model
    agent = ToTAgent(model_name="gpt2", max_depth=3, num_thoughts=5, temperature=0.7)

    # Example prompt
    prompt = "What is the best way to learn programming?"

    # Generate response using Tree of Thoughts
    response = agent.generate_response(prompt, max_tokens=100)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
