from config_to_object import load_config
from bilstm.config_types import Config
import os,sys

HOME_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")
config:Config = load_config("config.ini")

def to_html(text,labels,predictions,probs,out_file,template_location,css_location):
    with open(template_location,"r") as f:
        template = f.read()

    inside = ""
    for t,l,pre,pro in zip(text,labels,predictions,probs):
        inside+='<div class="container">\n'
        if l==pre:
            color_class = "green"
        else:
            color_class = "red"
        inside+=f'<div class="container_elem {"sponsor" if l=="sponsor" else ""}">{l}</div>\n'
        inside+=f'<div class="container_elem {color_class}">{round(pro,2)} {pre}</div>\n'
        inside+=f'<div class="container_elem">{t}</div>\n'
        inside+='</div>\n'
    
    template = template.replace("{CONTENT}",inside).replace("{CSS_PATH}",os.path.relpath(css_location,os.path.dirname(out_file)))

    with open(out_file,"w") as f:
        f.write(template)