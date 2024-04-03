from langchain.tools import ShellTool
from pydantic import BaseModel, Field


def shell(query: str):
    tool = ShellTool()
    return tool.run(tool_input=query)

class ShellInput(BaseModel):
    query: str = Field(description="Một lệnh Shell có thể chạy trên dòng lệnh Linux")
