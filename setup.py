from setuptools import setup, find_packages
setup(
    name="macro-regime-hmm",
    version="0.1.0",
    description="Gaussian HMM for macro regime identification. Baum-Welch EM + Viterbi from scratch. Regime-conditional factor allocation.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
