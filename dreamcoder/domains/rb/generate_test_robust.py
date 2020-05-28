import random

# ========================== ADDRESS ==============================
def street_number():
    return random.choice([_ for _ in range(400)])

def street():
    return random.choice(['Pine', "Brook", "Foothill", "Dew Point", "First", "Main", "Wood Violet", "Log Pond", "Fourth", "Keystone", "Evergreen"])

def street_type():
    return random.choice(["Street", "Road", "Way", "Ave", "St", "Lane"])

def street_zip():
    return random.choice( range(10000, 100000))

def street_state():
    return random.choice(['CA', 'PA', 'IL', 'NY', 'MA', 'AK', 'WY', 'MD', 'LA', 'KY'])

def address_info():
    return street_number(), street(), street_type(), street_zip(), street_state()

def street_ex1(st_num, st, st_type, st_zip, st_state):
    return [f"{st_num} {st} {st_type}", f"Street:{st}, House num:{st_num} "]
def street_ex2(st_num, st, st_type, st_zip, st_state):
    return [f"{st_num} {st} {st_type}, {st_state}", f"{st} {st_type} ({st_state})"]
def street_ex3(st_num, st, st_type, st_zip, st_state):
    return [f"{st_num} {st} {st_type}, {st_state} {st_zip}", f"{st}, ({st_state} {st_zip})"]
def street_ex4(st_num, st, st_type, st_zip, st_state):
    zip_suffix = street_zip()
    return [f"{st_num}, {st} {st_type}, {st_state}, {st_zip}-{zip_suffix}", f"{st_state} {st_zip}"]

def street_ex(n_io):
    tasks = []
    for fn in [street_ex1, street_ex2, street_ex3, street_ex4]:
        task = None
        while not task:
            infos = [address_info() for _ in range(n_io)]
            task = list(zip(*[fn(*info) for info in infos]))
            ins, outs = task
            if any( len(i) > 36 for i in ins) or any( len(o) > 36 for o in outs):
                task = None
        tasks.append( task )
    return tasks

# ========================== REVIEWS ==========================
def review_user():
    return f"{names_first()[0]}. {names_last()[0]}."

def review_date():
    return f"{month()}/{year()}"

def review_content():
    return random.choice(["good", 
                          "liked it",
                          "I farted",
                          "Fun for all",
                          "great fun!",
                          "boring",
                          "exciting",
                          "not so fun",
                          ])
def review_id():
    return random.choice(range(1000, 10000-1))

def review_score():
    return random.choice([1,2,3,4,5])

def review_info():
    return review_user(), review_id(), review_date(), review_content(), review_score()

def review_ex1(user, id, date, content, score):
    return [f"{user} {date}: {content}, {score}", f"{content}"]

def review_ex2(user, id, date, content, score):
    return [f"{user} [{id}] comment: {content}, {score}", f"{user} said: {content}"]

def review_ex3(user, id, date, content, score):
    return [f"{user} @{id} '{content}' {score}", f"{id} [{content}] {score}"]

def review_ex4(user, id, date, content, score):
    return [f"{user} {id} '{content}' on {date}", f"{date} {id}"]

def review_ex5(user, id, date, content, score):
    return [f"on {date} {user}: {content}", f"{content}"]

def review_ex(n_io):
    tasks = []

    for fn in [review_ex1, review_ex2, review_ex3, review_ex4, review_ex5]:
        task = None
        while not task:
            infos = [review_info() for _ in range(n_io)]
            task = list(zip(*[fn(*info) for info in infos]))
            ins, outs = task
            if any( len(i) > 36 for i in ins) or any( len(o) > 36 for o in outs):
                task = None
        tasks.append( task )

    return tasks

# ========================= NAMES ============================
def names_first():
    return random.choice(["Alex", "Sasha", "Taylor", "Jackie", "Isaac", "Norman", "Sarah Lee", "Mary Jane"])

def names_last():
    return random.choice(["Lennon", "Smith", "Einstein", "Schmidhuber", "MacDonald", "McCormick"]) 

def names_middle():
    return random.choice([names_last(), ""])

def names_title():
    return random.choice(["Dr", "Sir", "Mr", "Mrs", "Miss"])

def names_suffix():
    return random.choice([random.choice(["Esq", "I", "III", "Jr."]), ""])

def name_info():
    return names_first(), names_last(), names_middle(), names_title(), names_suffix()

def name_ex1(first, last, middle, title, suffix):
    return [f"{title} {first} {middle} {last} {suffix}", f"{first} : {last}"]
def name_ex2(first, last, middle, title, suffix):
    return [f"{first} {middle} {last}", f"Dr {last}"]
def name_ex3(first, last, middle, title, suffix):
    return [f"{title} {first} {middle} {last}", f"{last}, {first} ({title})"]
def name_ex4(first, last, middle, title, suffix):
    last_letter = last[0]
    first_letter = first[0]
    return [f"{title} {first} {middle} {last}", f"{last}, {first_letter})"]

def name_ex(n_io):
    tasks = []
    for fn in [name_ex1, name_ex2, name_ex3, name_ex4]:
        task = None
        while not task:
            infos = [name_info() for _ in range(n_io)]
            task = list(zip(*[fn(*info) for info in infos]))
            ins, outs = task
            if any( len(i) > 36 for i in ins) or any( len(o) > 36 for o in outs):
                task = None
        tasks.append( task )
    return tasks

########Phone number stuff
def threeD():
    return "".join(str(d) for d in random.choices(range(10), k=3))

def fourD():
    return "".join(str(d) for d in random.choices(range(10), k=4))

def phone_type():
    return random.choice(["home", "work", "cell"])

def phone_info():
    return threeD(), threeD(), fourD(), phone_type()

def phone_ex1(area, three, four,  phone_type ):
    return [f"{area}{three}{four}", f"({area}) {three}-{four}"] 

def phone_ex2(area, three, four,  phone_type):
    return [f"({area}) {three} {four}", f"area code: {area}, num: {three}{four}"] 

def phone_ex3(area, three, four,  phone_type):
    return [f"{phone_type}: {area} {three}{four}", f"(+{area}) {three}-{four}, type: {phone_type}"]

def phone_ex4(area, three, four, phone_type ):
    return [f"{area}-{three}-{four}", f"({area}) {three}{four}"]

def phone_ex5(area, three, four, phone_type):
    return [f"{phone_type}: {area}-{three}-{four}", f"({area}) {three}{four} ({phone_type})"] 

def phone_ex6(area, three, four,  phone_type):
    return [f"{phone_type}: {area}{three}{four}", f"({area}) {three}{four}, type={phone_type}"]


def phone_ex(n_io):
    tasks = []
    for fn in [phone_ex1, phone_ex2, phone_ex3, phone_ex4, phone_ex5, phone_ex6]:
        task = None
        while not task:
            infos = [phone_info() for _ in range(n_io)]
            task = list(zip(*[fn(*info) for info in infos]))
            ins, outs = task
            if any( len(i) > 36 for i in ins) or any( len(o) > 36 for o in outs):
                task = None
        tasks.append( task )
    return tasks



####### date and time ######
def day():
    return random.choice( range(1, 32) )

def weekday():
    return random.choice(["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"])

def month():
    return random.choice(["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"] )

def year():
    return random.choice( range(1970, 2030) )

def hours():
    return random.choice(range(1,13))

def minute():
    return str(random.choice(range(6))) + str(random.choice(range(10)))

def AP():
    return random.choice(["AM", "PM"])


def date_info():
    return day(), weekday(), month(), year(), hours(), minute(), AP()

def date_ex1(day, weekday, month, year, hour, minute, AP):
    return [f"{weekday}, {month} {day} {hour}:{minute}", f"{month} {day} at {hour} o'clock"]

def date_ex2(day, weekday, month, year, hour, minute, AP):
    return [f"{weekday}, {month} {day}, {hour}:{minute}{AP}", f"{weekday} at approx. {hour} {AP}"]

def date_ex3(day, weekday, month, year, hour, minute, AP):
    return [f"{day} {month} {year}", f"year: {year}; month: {month}"]

def date_ex4(day, weekday, month, year, hour, minute, AP):
    return [f"{day} {month} {year}", f"{month} {day}"]

def date_ex5(day, weekday, month, year, hour, minute, AP):
    return [f"{day}-{month} ({weekday})", f"{weekday} ({month} {day})"]

def date_ex6(day, weekday, month, year, hour, minute, AP):
    return [f"{day}-{hour}-{year}", f"{hour}/{day}/{year}"]

def date_ex7(day, weekday, month, year, hour, minute, AP): #too hard
    return [f"date: {day} mo: {hour} year: {year}", f"{hour}/{day}/{year}"]

def date_ex8(day, weekday, month, year, hour, minute, AP): #too hard
    return [f"{hour}/{day}/{year}", f"date: {day} mo: {hour} year: {year}"]

def date_ex9(day, weekday, month, year, hour, minute, AP):
    return [f"{weekday}, {month} {day}, {hour}:{minute} {AP}", f"{weekday} at {hour} {AP}"]

def date_ex10(day, weekday, month, year, hour, minute, AP):
    return [f"{month} {day}, {hour}:{minute} {AP}", f"{month} {day}, approx. {hour} {AP}"]

def date_ex11(day, weekday, month, year, hour, minute, AP): #bad
    return [f"{month} {day}, {hour}:{minute} {AP}", f"{month} {day}, at {hour}:{minute} ({AP})"]

def date_ex12(day, weekday, month, year, hour, minute, AP):
    return [f"{month} {day}, {hour}:{minute} {AP}", f"{hour}:{minute} ({AP}) on {month} {day}"]

def date_ex12(day, weekday, month, year, hour, minute, AP):
    return [f"{month} {day}, {hour}:{minute} {AP}", f"{month} {day} (at {hour}:{minute} {AP}) "]

def date_ex13(day, weekday, month, year, hour, minute, AP):
    mo=hours()
    return [f"{day}/{mo}, {hour}:{minute} {AP}", f"{mo}/{day}, at {hour}:{minute} ({AP})"]

def date_ex14(day, weekday, month, year, hour, minute, AP):
    mo=hours()
    return [f"{day}/{mo}, {hour}:{minute} {AP}", f"{hour}:{minute} ({AP}) on {month} {day}"]

def date_ex15(day, weekday, month, year, hour, minute, AP):
    mo=hours()
    return [f"{day}/{mo}, {hour}:{minute} {AP}", f"{mo}/{day} (at {hour}:{minute}  {AP}) "]


def date_ex(n_io):
    tasks = []
    for fn in [date_ex1, date_ex2, date_ex3, date_ex4, date_ex5, date_ex6, 
                date_ex7, date_ex8, date_ex9, date_ex10, date_ex11, date_ex12, 
                date_ex13, date_ex14, date_ex15]:
        task = None
        while not task:
            infos = [date_info() for _ in range(n_io)]
            task = list(zip(*[fn(*info) for info in infos]))
            ins, outs = task
            if any( len(i) > 36 for i in ins) or any( len(o) > 36 for o in outs):
                task = None
        tasks.append( task )
    return tasks




if __name__ == '__main__':

    tasks = date_ex(4) + street_ex(4) + name_ex(4) + phone_ex(4) + review_ex(4) 
    tasks = tasks + date_ex(4) + street_ex(4) + name_ex(4) + phone_ex(4) + review_ex(4) 
    tasks = tasks + street_ex(4) + name_ex(4) + phone_ex(4) + review_ex(4)

    print(len(tasks))

    for ins, outs in tasks:
        assert not (any( len(i) > 36 for i in ins) or any( len(o) > 36 for o in outs))

    import dill
    with open("full_dataset.p", 'wb') as h:
        dill.dump(tasks, h)
