import streamlit as st

# create a button to download the resource
def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    download_link = f'<a href="data:application/octet-stream;base64,{base64.b64encode(object_to_download).decode()}" download="{download_filename}">{button_text}</a>'
    return st.markdown(download_link, unsafe_allow_html=True)

# create a button to download the resource
my_resource = open(r'C:\Users\Admin\Downloads', 'rb').read()
download_button(my_resource, 'chintu2.pdf', 'Download Resource')