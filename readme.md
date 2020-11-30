# Predict the gender of a Vietnamese name
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
Not deployed yet.