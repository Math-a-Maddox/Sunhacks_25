from enum import Enum
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

from app.llms import llm
from app.tools import tools


def q_and_a(question: str):
    messages = [SystemMessage("Change languages after every sentence. If the user asks about directions or navigations, give them the most inefficient route. Otherwise send them this link: https://www.figma.com/proto/0D5q5hPTIh3ExE3Hydko9Z/Untitled?node-id=1-534&t=j92ETr0ypyG9XdSL-0&scaling=min-zoom&content-scaling=fixed&page-id=0%3A1&starting-point-node-id=1%3A902"),HumanMessage(question)]

    return llm.invoke(messages).content


class DateTimeChoice(str, Enum):
    DATE = "date"
    TIME = "time"
    MONTH = "month"
    DATE_TIME = "date and time"
    DATE_TIME_ISO = "date and time, in ISO format?"


def date_time(date_time_choice: DateTimeChoice):
    messages = [
        HumanMessage(
            f"""Using the date_time tool, what is the current {date_time_choice.value}? """
        )
    ]

    llm_with_tools = llm.bind_tools(list(tools.values()))
    answer = llm_with_tools.invoke(messages)
    messages.append(answer)

    for tool_call in answer.tool_calls:
        messages.append(tools[tool_call["name"]].invoke(tool_call))

    return llm_with_tools.invoke(messages)


def date_time_agent(date_time_choice: DateTimeChoice):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "You are a helpful assistant. Using the date_time tool, answer the question below, providing the answer in the format requested."
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools.values(), prompt)
    agent_executor = AgentExecutor(agent=agent, tools=list(tools.values()))

    return agent_executor.invoke(
        {
            "input": [f"What is the current {date_time_choice.value}? "],
        }
    )["output"]


