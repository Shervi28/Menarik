import streamlit as st
import arxiv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser

api="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"

st.set_page_config(layout="wide")

def summarymaker(text):
        stopWords = set(stopwords.words("english"))
        words = word_tokenize(text)
        freqTable = dict() 
        for word in words:
            word = word.lower()
            if word in stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1
        
        sentences = sent_tokenize(text)
        sentenceValue = dict()
        
        for sentence in sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq
        sumValues = 0
        for sentence in sentenceValue:
            sumValues += sentenceValue[sentence]
        
        average = int(sumValues / len(sentenceValue))

        summary = ''
        for sentence in sentences:
            if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
                summary += " " + sentence
        out_file=open("summary_out.txt","w+")
        out_file.write(summary)
        return summary


st.title("Menarik 🔬🔍")
st.write("Demystify research topics")
topic = st.text_input("Enter the main topic you are interested in:")



papers = {}

search = arxiv.Search(
query = topic,
max_results = 5,
sort_by = arxiv.SortCriterion.SubmittedDate
)

for result in arxiv.Client().results(search):
    papers[result.title] = [result.entry_id, result.summary, result.pdf_url]

llm2 = OpenAI(openai_api_key=api, temperature=0.7)
output_parser2 = CommaSeparatedListOutputParser()
template2 = """You give a concise and easy to understand summary that a person could easily understand in 1 second
    Question: Give summary to: {text} 
    Answer: Keep it short and easy to understand.
    """

prompt_template2 = PromptTemplate(input_variables=["text"], template=template2, output_parser=output_parser2)
answer_chain2 = LLMChain(llm=llm2, prompt=prompt_template2)

for i in papers:
        st.subheader(i)
        st.caption("URL: " + papers[i][2])
        
        st.write("Feynman Bot's Summary: ", answer_chain2.run(papers[i][1]))
        st.write("Original Abstract: ", papers[i][1])
        st.divider()

llm1 = OpenAI(openai_api_key=api, temperature=0.7)
output_parser = CommaSeparatedListOutputParser()
template1 = """You answer questions based on a bunch of summaries of research papers we send you
    Question: Answer the following question: {text} based on {papers}
    Answer: Keep it short and easy to understand.
    """

prompt_template1 = PromptTemplate(input_variables=["text", "papers"], template=template1, output_parser=output_parser)
answer_chain1 = LLMChain(llm=llm1, prompt=prompt_template1)

st.header("Feynman Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
         st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        inputs = {
            'text': prompt,
            'papers': ''.join([summarymaker(papers[i][1]) for i in papers])
        }

        response = answer_chain1.run(inputs)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

