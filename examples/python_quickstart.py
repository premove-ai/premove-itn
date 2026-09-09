"""Load Premove once and normalize several complete transcripts."""

from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained()

texts = [
    "call me at four thirty",
    "the total is twenty dollars",
    "the last account digits are zero eight two zero six three",
]

for text in texts:
    print(itn.normalize(text))
