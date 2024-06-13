import streamlit as st
from langchain.document_loaders import CSVLoader
import tempfile
from utils import model_response



def main():

    st.title("Chat with CSV Using CSV file")

    ### File Uploader

    uploader_file=st.sidebar.file_uploader("Choose a CSV File",type='CSV')
    ### Fetch path of the Uploaded file 
    if uploader_file:
        ## Use a Temp file becaise CSV loader Accepts only File Path

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploader_file.getvalue())
            tmp_file_path=tmp_file.name

            csv_file=CSVLoader(file_path=tmp_file_path,encoding="utf-8",csv_args={"delimiter":","})

            data=csv_file.load()

            user_input=st.text_input("Your Message")

            if user_input:
                response=model_response(data,user_input)
               
                st.write(response)


        






if __name__=='__main__':

    main()
