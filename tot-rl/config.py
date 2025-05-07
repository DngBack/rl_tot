import os
import argparse

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Tree of Thoughts CLI")
        parser.add_argument('--backend', choices=['openai', 'huggingface'], default=os.getenv('TOT_BACKEND', 'openai'))
        parser.add_argument('--hf_model', type=str, default=os.getenv('HF_MODEL', 'facebook/opt-1.3b'))
        parser.add_argument('--task', type=str, required=True, help='Name of the task to run')
        parser.add_argument('--max_depth', type=int, default=3)
        parser.add_argument('--beam_width', type=int, default=5)
        parser.add_argument('--openai_api_key', type=str, default=os.getenv('OPENAI_API_KEY', ''))
        self.args = parser.parse_args()
        self.backend = self.args.backend
        self.hf_model = self.args.hf_model
        self.task = self.args.task
        self.max_depth = self.args.max_depth
        self.beam_width = self.args.beam_width
        self.openai_api_key = self.args.openai_api_key
