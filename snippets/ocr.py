import easyocr
reader = easyocr.Reader(['en'])
result = reader.readtext('1.jpg', paragraph=True)
print(result)