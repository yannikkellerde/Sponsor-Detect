from config_to_object import load_config
from bilstm.config_types import Config
import os,sys

HOME_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")
config:Config = load_config("config.ini")

def to_html(text,labels,predictions,out_file,template_location,css_location):
    with open(template_location,"r") as f:
        template = f.read()

    inside = ""
    for t,l,p in zip(text,labels,predictions):
        inside+='<div class="container">\n'
        if l==p:
            color_class = "green"
        else:
            color_class = "red"
        inside+=f'<div class="container_elem">{l}</div>\n'
        inside+=f'<div class="container_elem">{p}</div>\n'
        inside+=f'<div class="container_elem {color_class}">{t}</div>\n'
        inside+='</div>\n'
    
    template = template.replace("{CONTENT}",inside).replace("{CSS_PATH}",os.path.relpath(css_location,os.path.dirname(out_file)))

    with open(out_file,"w") as f:
        f.write(template)