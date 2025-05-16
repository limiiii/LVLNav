#!/home/rob/anaconda3/envs/pytorch/bin/python
# -*- coding: utf-8 -*-

from openai import OpenAI


class DSserver1:
    def __init__(self):
        # 设置OpenAI API密钥
        self.client = OpenAI(api_key="sk-c87fd21390624d72a65c24e0cc795832", base_url="https://api.deepseek.com")
        self.user_input = "现在需要你按照如下的规则提取示例语句中的物品名称，仿照下面的四个回答的例子返回由物品名称组成的列表，注意你只需要返回列表即可,并且在列表中的每个物品可以重复出，但只允许用一个单词表述，不需要附带任何的说明！示例一：Pass the pool and go indoors using the double glass doors. Pass the large table with chairs and turn left and wait by the wine bottles that have grapes by them. 回答：['Pool', 'Door', 'Chair', 'Bottle'];示例二：Walk straight through the room and exit out the door on the left. Keep going past the large table and turn left. Walk down the hallway and stop when you reach the 2 entry ways. One in front of you and one to your right. The bar area is to your left.回答：['Room', 'Door', 'Table', 'Hallway', 'Bar'];示例三：Enter house through double doors, continue straight across dining room, turn left into bar and stop on the circle on the ground.回答：['Door', 'Dinning Room', 'Bar'];示例四：Standing in front of the family picture, turn left and walk straight through the bathroom past the tub and mirrors. Go through the doorway and stop when the door to the bathroom is on your right and the door to the closet is to your left.回答：['Family Picture', 'Bathroom', 'Tub', 'Mirrors', 'Door', 'Bathroom', 'Closet'];下面请你提取如下语句的关键物品，返回一个列表："
        print("chatgptserver started!")
    def chat_with_gpt(self, message):
        msgs = self.user_input + message
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": msgs},
            ],
            stream=False
        )

    # 提取ChatGPT的回复消息
        reply = response.choices[0].message.content
        print(type(reply))
        reply = eval(reply)
        return reply
