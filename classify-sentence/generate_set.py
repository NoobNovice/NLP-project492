import re
import sys
import random
from pythainlp import word_tokenize

Start = ("","","","","เพื่อน","มึง","เมิง","มึง ๆ","เอ็ง","นาย","เธอ","มึงๆ","เห้ย","","","","","","","","","")
Topic = ("อาหาร","เมนู","เมนูอาหาร")
Prop = ("อร่อย","น่าสนใจ","ดีๆ","ดี ๆ","อร่อย","น่ากิน","เด็ด","เด็ด ๆ","น่าทาน","ถูกๆ","","","","")
Time = ("เช้า","สาย","สาย ๆ","บ่าย","เย็น","วันนี้","ตอนนี้","มื้อนี้","เช้านี้","สายนี้","บ่ายนี้","เย็นนี้")
Date = ("จันทร์","อังคาร","พุธ","พฤหัสบดี","ศุกร์","เสาร์","อาทิตย์","พฤหัสบดี")
Clock = ("เที่ยงคืน","ตีหนึ่ง","ตีสอง","ตีสาม","ตีสี่","ตีห้า",
             "หกโมง","เจ็ดโมง","แปดโมง","เก้าโมง","สิบเอ็ดโมง","เที่ยง",
             "บ่าย","บ่ายสอง","บ่ายสาม","บ่ายสี่","สี่โมงเย็น","ห้าโมงเย็น","หกโมงเย็น"
             "ทุ่ม","สองทุ่ม","สามทุ่ม","สี่ทุ่ม","ห้าทุ่ม","เที่ยงคืนครึ่ง","ตีหนึ่งครึ่ง","ตีสองครึ่ง","ตีสามครึ่ง","ตีสี่ครึ่ง",
             "ตีห้าครึ่ง","หกโมงครึ่ง","เจ็ดโมงครึ่ง","แปดโมงครึ่ง","เก้าโมงครึ่ง","สิบเอ็ดโมงครึ่ง","เที่ยงครึ่ง",
             "บ่ายครึ่ง","บ่ายสองครึ่ง","บ่ายสามครึ่ง","บ่ายสี่ครึ่ง","สี่โมงเย็นครึ่ง","ห้าโมงเย็นครึ่ง","หกโมงเย็นครึ่ง"
             "ทุ่มครึ่ง","สองทุ่มครึ่ง","สามทุ่มครึ่ง","สี่ทุ่มครึ่ง","ห้าทุ่มครึ่ง")
Noun = ("ไม้","จาน","ชาม","ถ้วย","หัว","แท่ง","ชิ้น","แก้ว")
Prep = ("ติดกับ","ถัดจาก","ใกล้","อยู่ใกล้กับ","ตรงข้าม","อยู่ตรงข้าม","ใกล้ๆกับ","เยื้อง")
Build = ("สี่แยกไฟแดง","บิกซี","เซเว่น","เทสโก้โลตัส","แม็คโคร","7-11","ถนนไฮเวย์")
Num = ("1","2","3","4","5","6","7","8","9","0")
Type = ()
Location = ()
Name = ()

if __name__ == "__main__":
    template = []
    amount_gen = int(sys.argv[3])
    with open(sys.argv[1], "r", encoding='utf-8-sig') as f:
        while True:
            temp = f.readline()
            if not temp:
                break
            temp = temp.replace("\n", '')
            template.append(temp)
        f.close()

    temp_list = []
    with open("Type.txt", "r", encoding='utf-8-sig') as f:
        while True:
            temp = f.readline()
            if not temp:
                break
            temp = temp.replace("\n", '')
            temp_list.append(temp)
        f.close()
    Type = tuple(temp_list)

    temp_list = []
    with open("Location.txt", "r", encoding='utf-8-sig') as f:
        while True:
            temp = f.readline()
            if not temp:
                break
            temp = temp.replace("\n", '')
            temp_list.append(temp)
        f.close()
    Location = tuple(temp_list)

    temp_list = []
    with open("Name.txt", "r", encoding='utf-8-sig') as f:
        while True:
            temp = f.readline()
            if not temp:
                break
            temp = temp.replace("\n", '')
            temp_list.append(temp)
        f.close()
    Name = tuple(temp_list)

    gen_sentence = []
    for i in range(amount_gen):
        sentence = template[random.randint(0,len(template)-1)]
        sentence = Start[random.randint(0,len(Start)-1)] + sentence
        if re.search("Topic", sentence):
            sentence = re.sub("Topic", Topic[random.randint(0,len(Topic)-1)], sentence)
        if re.search("Prop", sentence):
            sentence = re.sub("Prop", Prop[random.randint(0,len(Prop)-1)], sentence)
        if re.search("Time", sentence):
            sentence = re.sub("Time", Time[random.randint(0,len(Time)-1)], sentence)
        if re.search("Noun", sentence):
            sentence = re.sub("Noun", Noun[random.randint(0,len(Noun)-1)], sentence)
        if re.search("Prep", sentence):
            sentence = re.sub("Prep", Prep[random.randint(0,len(Prep)-1)], sentence)
        if re.search("Build", sentence):
            sentence = re.sub("Build", Build[random.randint(0,len(Build)-1)], sentence)
        if re.search("Location", sentence):
            sentence = re.sub("Location", Location[random.randint(0,len(Location)-1)], sentence)
        if re.search("Type", sentence):
            sentence = re.sub("Type", Type[random.randint(0,len(Type)-1)], sentence)
        
        token = word_tokenize(sentence, engine='newmm')
        while "Name" in token:
            index = token.index("Name")
            token[index] = Name[random.randint(0,len(Name)-1)]
        while "Num" in token:
            index = token.index("Num")
            token[index] = Num[random.randint(0,len(Num)-1)]
        while "Clock" in token:
            index = token.index("Clock")
            token[index] = Clock[random.randint(0,len(Clock)-1)]
        while "Date" in token:
            index = token.index("Date")
            token[index] = Date[random.randint(0,len(Date)-1)]
        sentence = ''.join([t for t in token if t != " "])
        gen_sentence.append(sentence)

    with open(sys.argv[2],"w", encoding='utf-8') as f:
        for d in gen_sentence:
            f.write(d+'\n')
        f.close