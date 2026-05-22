from setuptools import setup, find_packages

setup(
    name="lc-coder-agent",
    version="0.1.0",
    description="Fine-tuning Qwen2.5-7B for LeetCode code reasoning via SFT + GRPO",
    author="HKUST SuperPOD Team",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "loguru>=0.7.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
)
