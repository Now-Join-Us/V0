from huggingface_hub import hf_hub_download
from v0_core.models.v0 import V0

# --- Configuration ---
REPO_ID = "Now-Join-Us/Generalist-Value-Model-V0"

CHECKPOINT_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="v_0_for_grpo_training.pt"
)

TABPFN_MODEL_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="pretrained/tabpfn-v2.5-classifier-v2.5_default.ckpt"
)

EMBEDDING_MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B"

print(f"Loaded V0 Checkpoint from: {CHECKPOINT_PATH}")
print(f"Loaded TabPFN Head from: {TABPFN_MODEL_PATH}")

# --- Model Initialization ---
model = V0.from_pretrained(
    checkpoint_path=CHECKPOINT_PATH,
    embedding_model_path=EMBEDDING_MODEL_PATH,
    tabpfn_head_path=TABPFN_MODEL_PATH,
    device="cuda"
)

# --- Data Preparation ---
# Suggestion: For better performance, a larger and more diverse context is recommended (standard 256).
context = [
    {"prompt": "Three builders are scheduled to build a house in 60 days. However, they procrastinate and do nothing for the first 50 days. To complete the house on time, they decide to hire more workers and work at twice their original speed. If the new workers also work at this doubled rate, how many new workers are needed? Assume each builder works at the same rate and does not interfere with the others.", "is_correct": True},
    {"prompt": "Let $d(m)$ denote the number of positive integer divisors of a positive integer $m$. If $r$ is the number of integers $n \\leq 2023$ for which $\\sum_{i=1}^{n} d(i)$ is odd, find the sum of the digits of $r$.", "is_correct": True},
    {"prompt": "设在 $5 \\times 5$ 的方格表的第 $i$ 行第 $j$ 列所填的数为 $a_{i j}\\left(a_{i j} \\in\\{0,1\\}\\right), a_{i j}=a_{j i}(1 \\leqslant i、j \\leqslant 5)$ .则表中共有五个 1 的填表方法总数为 $\\qquad$ （用具体数字作答).", "is_correct": True},
    {"prompt": "Suppose $x, y \\in \\mathbb{Z}$ satisfy the equation:\n\\[\ny^4 + 4y^3 + 28y + 8x^3 + 6y^2 + 32x + 1 = (x^2 - y^2)(x^2 + y^2 + 24).\n\\]\nFind the sum of all possible values of $|xy|$.", "is_correct": False},
    {"prompt": "Let $P_0 = (3,1)$ and define $P_{n+1} = (x_n, y_n)$ for $n \\ge 0$ by \\[ x_{n+1} = - \\frac{3x_n - y_n}{2}, \\quad y_{n+1} = - \\frac{x_n + y_n}{2} \\] Find the area of the quadrilateral formed by the points $P_{96}, P_{97}, P_{98}, P_{99}$.", "is_correct": False}
]

target_prompts = [
    {"prompt": "Determine the largest integer $N$ for which there exists a $6 \\times N$ table $T$ that has the following properties:\n\n- Every column contains the numbers $1, 2, \\ldots, 6$ in some ordering.\n- For any two columns $i \\ne j$, there exists a row $r$ such that $T(r,i) = T(r,j)$.\n- For any two columns $i \\ne j$, there exists a row $s$ such that $T(s,i) \\ne T(s,j)$.", "is_correct": True},
    {"prompt": "What is the largest $n$ such that there exists a non-degenerate convex $n$-gon where each of its angles is an integer number of degrees, and all angles are distinct?", "is_correct": True},
    {"prompt": "已知四面体 \\(A B C D\\) 内接于球 \\(O\\)，且 \\(A D\\) 是球 \\(O\\) 的直径。若 \\(\\triangle A B C\\) 和 \\(\\triangle B C D\\) 都是边长为 1 的等边三角形，则四面体 \\(A B C D\\) 的体积是多少？原始答案的形式为 \\(\\frac{\\sqrt{c}}{b}\\)，请给出a+b+c的值。", "is_correct": False}
]

print(f">>> Running V0 on {len(target_prompts)} targets using {len(context)} context prompts...")

# --- Execution ---
scores = model.predict(
    context_prompts=[item['prompt'] for item in context],
    context_labels=[1 if item['is_correct'] else 0 for item in context],
    target_prompts=[item['prompt'] for item in target_prompts]
)

# --- Output Results ---
for i, (item, score) in enumerate(zip(target_prompts, scores)):
    print(f"Target prompt: {item['prompt'][:100]}...")
    print(f"Predicted Value Score: {score:.4f}; Ground Truth Label (is_correct): {item['is_correct']}")
    print("-" * 10)
