# Predict the gender of a Vietnamese name
LSTM version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16dUaD15aL86htijqE6hlBvf6KiJ26RMU)
BERT version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-XxeLfPBermYxKJxeQNHMsTCdllGFmQg)
The version of the deployed model is a two-layer bidirectional LSTM.
Send a POST request to https://gender-regconition.herokuapp.com/gender with only one parameter named `names` which contains names separated by a comma. The response is the probability of the name being male or female.
Example: https://gender-regconition.herokuapp.com/gender?names=quốc khánh,nguyễn thanh nhàn,hương,lan hương
Response:
```json
{
    "result": [
        [
            "quốc khánh",
            "nam",
            99.99995422363281
        ],
        [
            "nguyễn thanh nhàn",
            "nữ",
            99.87543487548828
        ],
        [
            "hương",
            "nam",
            61.728614807128906
        ],
        [
            "lan hương",
            "nữ",
            97.12590789794922
        ]
    ]
}
```
# Generate names
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10TScnpluI6Cgfw3wqb0moDWxHXxRFvD3)
Character level base, four layers LSTM.
Here a some generated names by the model with seed `b`:
>bùi chí công<pad>
bùi kiên giang<pad>
bùi lọ khuê<pad>
bùi sơn lâm<pad>
bùi bảo lễ<pad>
bùi ngọc khanh<pad>
bùi thanh hoa<pad>
bùi văn nhật<pad>
bùi thành nhung<pad>
bùi hoài giang<pad>

The model has not deployed yet.