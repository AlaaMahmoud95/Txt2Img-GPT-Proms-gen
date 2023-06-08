import streamlit as st
import cv2 as cv
import time
import torch
from diffusers import StableDiffusionPipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def create_model(loc = "stabilityai/stable-diffusion-2-1-base", mch = 'cpu'):
    pipe = StableDiffusionPipeline.from_pretrained(loc)
    pipe = pipe.to(mch)
    return pipe


def tok_mod():
  tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  model = GPT2LMHeadModel.from_pretrained('FredZhang7/distilgpt2-stable-diffusion-v2')
  model.to('cpu')
  return model,tokenizer


t2i = st.title("""
Txt2Img
###### `CLICK "Create_Update_Model"` :
- `FIRST RUN OF THE CODE`
- `CHANGING MODEL`
###### TO USE GPT PROMPTS GENERATOR CHECK `GPT PROMS` THEN CLICK `CREATE GPT MODEL`""")

the_type = st.selectbox("Model",("stabilityai/stable-diffusion-2-1-base",
                                      "CompVis/stable-diffusion-v1-4"))
st.session_state.gate = False

ma_1,_,ma_2 = st.columns([2,2,2])

with ma_1 :
  create = st.button("Create The Model")

if create:
    st.session_state.t2m_mod = create_model(loc=the_type)

with ma_2 : 
  gpt = st.checkbox("GPT PROMS")

if gpt :
    gen = st.button("Create GPT Model")
    if gen:
        st.session_state.mod,st.session_state.tok = tok_mod()

    m1,m2,m3 = st.columns([1,1,3])
    m4,m5 = st.columns(2)
    prompt = st.text_input("GPT PROM",r'' )
    with m1 :
      temperature = st.slider("Temp",0.0,1.0,.9,.1)   
    with m2 :        
      top_k = st.slider("K",2,16,8,2)  
    with m3 :                                               
      max_length = st.slider("Length",10,100,80,1)                                        
    with m4 :
      repitition_penalty = st.slider("penality",1.0,5.0,1.2,1.0)                                
    with m5 :
      num_return_sequences=st.slider("Proms Num",1,10,5,1)

    prom_gen = st.button("Generate Proms")

    if prom_gen :
        model, tokenizer = st.session_state.mod,st.session_state.tok
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        output = model.generate(input_ids, do_sample=True, temperature=temperature, top_k=top_k, max_length=max_length,
                                num_return_sequences=num_return_sequences, repetition_penalty=repitition_penalty,
                                penalty_alpha=0.6, no_repeat_ngram_size=1, early_stopping=True)

        st.session_state.PROMPTS = []
        for i in range(len(output)):
          st.session_state.PROMPTS.append(tokenizer.decode(output[i]))

if 'PROMPTS' in st.session_state :
    prom = st.selectbox("Proms",st.session_state.PROMPTS)

else :
    prom = st.text_input("# Prompt",'')



                               
c1,c2,c3 = st.columns([1,1,3])
c4,c5 = st.columns(2)
with c1:
  bu_1 = st.text_input("Seed",'999')
with c2:
  bu_2 = st.text_input("Steps",'12')
with c3:
  bu_3 = st.text_input("Number of Images",'1')
with c4:
  sl_1 = st.slider("Width",128,1024,512,8)
with c5:
  sl_2 = st.slider("hight",128,1024,512,8)

st.session_state.generator = torch.Generator("cpu").manual_seed(int(bu_1))

create = st.button("Imagine")

if create:
    model = st.session_state.t2m_mod
    generator = st.session_state.generator

    if int(bu_3) == 1 :
      IMG = model(prom, width=int(sl_1), height=int(sl_2),
                    num_inference_steps=int(bu_2),
                    generator=generator).images[0]
      st.image(IMG)
        
    else :
      PROMS = [prom]*int(bu_3)
        
      IMGS = model(PROMS, width=int(sl_1), height=int(sl_2),
                     num_inference_steps=int(bu_2),
                     generator=generator).images
    
      st.image(IMGS)
