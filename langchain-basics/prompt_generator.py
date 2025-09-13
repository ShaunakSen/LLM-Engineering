from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    template="""
    You are a helpful assistant that can generate a short report on the topic: {paper_input}
    The report should be in the style of {style_input} and the length should be {length_input}
    """,
    input_variables=["paper_input", "style_input", "length_input"]
)

template.save("./prompt_templates/template.json")