import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 设置设备 (GPU优先)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 加载专家模型和业余模型
# 专家模型 (Expert Model): 使用一个更大的模型，例如 gpt2-large
print("正在加载专家模型 (gpt2-large)...")
expert_model = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device)

# 业余模型 (Amateur Model): 使用一个较小的模型，例如 gpt2
print("正在加载业余模型 (gpt2)...")
amateur_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# 3. 加载分词器 (两个模型使用相同的分词器)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 4. 准备输入文本
prompt_text = "The quick brown fox jumps over the lazy dog and"
input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

# 5. 设置生成参数
max_length = 50
alpha = 0.5 # 对比解码的关键参数，控制业余模型的惩罚权重

# 6. 开始生成循环
print("\n--- 开始对比解码生成 ---")
print(f"初始提示: {prompt_text}")

generated_ids = input_ids

with torch.no_grad():
    for _ in range(max_length - input_ids.shape[1]):
        # a. 分别获取两个模型的 logits (原始预测分数)
        # `[:, -1, :]` 获取最后一个token的logits
        # [batch_size, sequence_length, vocab_size]
        # 这个切片操作将 [batch_size, sequence_length, vocab_size] 形状的张量
        # 变成了 [batch_size, vocab_size] 形状的二维张量
        expert_logits = expert_model(generated_ids).logits[:, -1, :]
        amateur_logits = amateur_model(generated_ids).logits[:, -1, :]

        # b. 计算 log softmax
        # 这一步将logits转换为对数概率，更利于数值计算
        expert_log_probs = F.log_softmax(expert_logits, dim=-1)
        amateur_log_probs = F.log_softmax(amateur_logits, dim=-1)

        # c. 计算对比分数 (Contrastive Scores)
        # scores = log_prob_expert - alpha * log_prob_amateur
        contrastive_scores = expert_log_probs - alpha * amateur_log_probs

        # d. 贪婪地选择分数最高的token作为下一个token
        next_token_id = torch.argmax(contrastive_scores, dim=-1).unsqueeze(0)

        # 检查是否生成了 EOS (End-of-Sentence) token
        if next_token_id == tokenizer.eos_token_id:
            break

        # e. 将新的token添加到已生成的序列中
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

# 7. 解码并打印最终结果
final_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\n--- 对比解码生成结果 ---")
print(final_text)
