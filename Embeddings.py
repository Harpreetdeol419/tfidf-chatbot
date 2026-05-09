import numpy as np

lr = 0.05 ##

vocab = [
    "hi", "hello", "hey", "yo", "bye", "goodbye", "thanks", "thank", "welcome", "sorry",
    "yes", "no", "ok", "okay", "maybe", "help", "please", "code", "bug", "fix",
    "error", "learn", "teach", "explain", "understand", "what", "why", "how", "when", "where",
    "who", "python", "numpy", "array", "vector", "embedding", "model", "train", "epoch", "weight",
    "bias", "loss", "input", "output", "data", "text", "file", "system", "search", "find",
    "run", "create", "delete", "update", "check", "test", "good", "bad", "great", "awesome",
    "cool", "fast", "slow", "easy", "hard", "simple", "complex", "chat", "ai", "bot",
    "response", "question", "answer", "learned", "knowledge", "brain", "memory", "logic", "math", "codebot",
    "debug", "print", "loop", "function", "class", "variable", "string", "number", "true", "false",
    "start", "stop", "continue", "process", "network",
    "worst", "hate", "dislike", "fail", "failure", "broken", "crash", "problem", "issue"
]

pos = [
    ("good", "great"), ("great", "awesome"), ("happy", "joy"), ("smart", "intelligent"),
    ("fast", "quick"), ("easy", "simple"), ("love", "like"), ("help", "support"),
    ("learn", "teach"), ("success", "win")
]

neg = [
    ("bad", "worst"), ("error", "bug"), ("hate", "dislike"), ("slow", "lag"),
    ("fail", "failure"), ("broken", "crash"), ("sad", "angry"), ("wrong", "incorrect"),
    ("confused", "hard"), ("problem", "issue")
]

def WordToIndex(v):
    return {w:i for i,w in enumerate(v)}

wti = WordToIndex(vocab)
np.random.seed(1)
embeddings = np.random.randn(len(vocab), 2)*0.1

for epoch in range(2000):
    for a,b in pos:
        if a in wti and b in wti: ## postive emb
            i,j = wti[a],wti[b]
            diff = embeddings[i]-embeddings[j]
            embeddings[i]-=lr*diff
            embeddings[j]+=lr*diff

    for a,b in neg: ## negative emb
        if a in wti and b in wti:
            i,j = wti[a],wti[b]
            diff = embeddings[i]-embeddings[j]
            embeddings[i]+=lr*diff
            embeddings[j]-=lr*diff
            
def sim(a, b): ## Get Similarity
    if a in wti and b in wti:
        i,j = wti[a], wti[b]
        v1,v2 = embeddings[i],embeddings[j]
        return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        
print(sim("good","awesome")) ## Call sim to check similarity between too words