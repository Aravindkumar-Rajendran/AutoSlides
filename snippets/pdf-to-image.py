from pdf2image import convert_from_path, convert_from_bytes
images = convert_from_path('science.pdf', first_page=208, last_page=210, fmt="jpeg")
for i, im in enumerate(images):
    im.save(f'{str(i)}.jpg')