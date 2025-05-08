from rl_tot.core.agent import RLToT


def main():
    # Initialize agent
    agent = RLToT(model_name="gpt2", max_depth=3, num_thoughts=5, temperature=0.7)

    # Example training data
    training_data = [
        {
            "prompt": "What is the best way to learn programming?",
            "target": "Start with Python basics and build projects",
        },
        {
            "prompt": "How to improve coding skills?",
            "target": "Practice daily and contribute to open source",
        },
        {
            "prompt": "What makes a good programmer?",
            "target": "Problem-solving skills and continuous learning",
        },
    ]

    # Train the agent
    metrics = agent.train(
        training_data=training_data, num_epochs=5, batch_size=2, learning_rate=1e-5
    )

    print("Training metrics:", metrics)

    # Test the trained agent
    test_prompt = "What is the best way to learn programming?"
    response = agent.generate(test_prompt, max_tokens=100)
    print(f"\nTest prompt: {test_prompt}")
    print(f"Response: {response[0]}")


if __name__ == "__main__":
    main()
