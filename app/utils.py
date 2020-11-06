import math
from app.Model import LSTM_classifier
import torch
import os

root_dir = os.getcwd()
model_folder = f'app/static/rnn-character-level'
print(f'loading model from: {model_folder}')
cpu = torch.device('cpu')

model = LSTM_classifier(hidden_size=256)
model.load_state_dict(torch.load(model_folder, map_location=cpu))
max_length = 25

alpha_bet = ' aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ'
char_to_idx = { char: index for index, char in enumerate(alpha_bet) }
idx_to_char = { index: char for index, char in enumerate(alpha_bet) }

end_token = '<end>'
end_token_id = 187
pad_token = '<pad>'
pad_token_id = 188

char_to_idx[end_token] = end_token_id
idx_to_char[end_token_id] = end_token
char_to_idx[pad_token] = pad_token_id
idx_to_char[pad_token_id] = pad_token

n_letters = 189

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
  try:
    return char_to_idx[letter]
  except:
    return -1;

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
  tensor = torch.zeros(n_letters)
  tensor[0][letterToIndex(letter)] = 1
  return tensor

# Turn a line into a <max_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
  tensor = torch.zeros(max_length, n_letters)
  for li, letter in enumerate(line):
    tensor[li][letterToIndex(letter)] = 1
  for index in range(len(line), max_length):
    tensor[index][pad_token_id] = 1
  return tensor

def test_model(names):
  results = []
  for name in names:
    input_test = lineToTensor(name.lower())
    pad_len = max_length - input_test.shape[0]
    pad_tensor = torch.zeros(pad_len, n_letters)
    pad_tensor[:,pad_token_id] = 1
    input = torch.cat((input_test, pad_tensor), dim=0)
    input = input.unsqueeze(0)
    input = input.to(cpu)
    model.eval()
    with torch.no_grad():
      output = model(input)
      exp = torch.randn(1, 2).fill_(math.exp(1)).to(cpu)
      res = torch.pow(exp, output)
      maxIdx = torch.argmax(res)
      gender = 'nữ' if maxIdx == 0 else 'nam'
      confidence = res[0][maxIdx] * 100;
      results.append((name, gender, confidence.item()))
  return results