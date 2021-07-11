import streamlit as st
import pandas as pd
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path, convert_from_bytes
import random
import shutil
import os
from autoslide import create_pptx
import base64
from subprocess import call


if not os.path.exists("images"): os.mkdir("images")


def read_pdf(file):
	pdfReader = PdfFileReader(file)
	count = pdfReader.numPages
	all_page_text = ""
	for i in range(count):
		page = pdfReader.getPage(i)
		all_page_text += page.extractText()
	return all_page_text


def convert_to_image(file, start_page, end_page):
    imgs = []
    images = convert_from_path(file, first_page=start_page, last_page=end_page, fmt="jpeg")
    for im in images:
        name = str(random.randint(0000, 9999))
        im.save(f'images/{name}.jpg')
        imgs.append(f'images/{name}.jpg')
    os.remove(file)
    return imgs


def main():
    st.title('AutoSlides - Create slides for teaching automagically')

    uploaded_file = st.file_uploader("Upload your textbook", type=["pdf"])
    start_page = int(st.number_input("start page", 1))
    end_page = int(st.number_input("end page", 2))

    if st.button("Create AutoSlides"):
        if uploaded_file is not None and start_page and end_page:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
            st.write("\n\nFile uploaded successfully")
            st.write(file_details)
            st.write("\n\n\n")
            if uploaded_file.type == "application/pdf":
				# extracted_text = read_pdf(uploaded_file)
                file_name = str(random.randint(0000, 9999)) + ".pdf"
                with open(file_name, "wb") as buffer:
                    shutil.copyfileobj(uploaded_file, buffer)
                images = convert_to_image(file_name, start_page, end_page)
                img_path, file_path = create_pptx(images)
                print(img_path)

                st.image(img_path)

                st.write("\n\n\n")
                st.write("Pdf :")
                
                call(["soffice", "--headless", "--convert-to", "pdf", file_path]) 

                pdf_file_path = os.path.splitext(os.path.split(file_path)[-1])[0] + ".pdf"
                with open(pdf_file_path,"rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="1000" height="1000" type="application/pdf">'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                
                with open(file_path,"rb") as f:
                    b64_pptx = base64.b64encode(f.read()).decode('utf-8')

                download_pptx = f'<a href="data:file/pptx;base64,{b64_pptx}" download="file.pptx">Download Slides</a>'    
                st.write("\n\n\n")
                st.markdown(download_pptx, unsafe_allow_html=True)


if __name__ == '__main__':
	main()