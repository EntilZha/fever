import toml


NOT_ENOUGH_INFO = "NOT ENOUGH INFO"
SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
VERIFIABLE = "VERIFIABLE"
NOT_VERIFIABLE = "NOT VERIFIABLE"


with open("serene.toml") as f:
    config = toml.load(f)
