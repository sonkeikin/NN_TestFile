import matplotlib.pyplot as plt
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

def point_append(add,train_point):
        if add not in train_point:
            train_point.append(add)
        elif add == 1:
            over_add = [num for num in train_point if 1<num<2]
            for i in over_add:
                train_point.remove(i)
            add_dx = 1 / (len(over_add) + 2) 
            add += add_dx
            while 1< add < 2-0.001:
                train_point.append(add)
                add += add_dx  
        elif add == 20:
            over_add = [num for num in train_point if 19<num<20]
            for i in over_add:
                train_point.remove(i)
            add_dx = 1 / (len(over_add) + 2) 
            add += add_dx - 1
            while 19< add < 20-0.001:
                train_point.append(add)
                add += add_dx
        else:
            over_add = [num for num in train_point if add - 1 <num< add + 1]
            for i in over_add:
                train_point.remove(i)
            train_point.append(add)
            print(len(over_add))
            if len(over_add) == 1:
                train_point.append(add - 0.5)
                train_point.append(add + 0.5)
            else:
                add_dx = 1 / ((len(over_add) - 1) / 2 + 2)

                add_minus = (add - 1 ) + add_dx
                while add - 1 < add_minus < add -0.001:
                    train_point.append(add_minus)
                    add_minus += add_dx

                add_plus = add + add_dx
                while add < add_plus < add + 1 -0.001:
                    train_point.append(add_plus)
                    add_plus += add_dx
        return train_point

def get_max_point(test_max_loss_list):
    tmp = {i:test_max_loss_list.count(i) for i in set(test_max_loss_list)}
    return max(zip(tmp.values(), tmp.keys()))[1]

def get_pic(content,time,label = "blank" , x_label = "x" , y_label = "y"):
    fig1 = plt.figure(1)
    x = np.arange(len(content))
    plt.plot(x, content, label='test max loss', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(x_label)
    plt.ylim(0, max(content))
    plt.legend(loc='upper right')
    label += str(time)
    plt.savefig('%s'%label , dpi=300) #指定分辨率保存 
    plt.draw()
    plt.pause(1)
    plt.close(fig1)

def sendmail(txt):
    smtpObj = smtplib.SMTP('smtp.gmail.com',587)
    smtpObj_SSL = smtplib.SMTP_SSL('smtp.gmail.com',465)
    smtpObj.ehlo()
    smtpObj_SSL.ehlo()
    smtpObj.starttls()
    smtpObj.login('sonkeikin@gmail.com','yy250520')
    smtpObj_SSL.login('sonkeikin@gmail.com','yy250520')
    msg = MIMEMultipart()
    # 设置邮件主题
    subject = Header('计算完成', 'utf-8').encode()
    msg['Subject'] = subject
    # 设置邮件发送者
    msg['From'] = 'sonkeikin@gmail.com <sonkeikin@gmail.com>'
    # 设置邮件接受者
    msg['To'] = '11615832@qq.com'
    # 添加文字内容
    text = MIMEText('%s'%txt, 'plain', 'utf-8')
    msg.attach(text)
    # 3.发送邮件
    smtpObj.sendmail('sonkeikin@gmail.com', '11615832@qq.com', msg.as_string())
    smtpObj.quit()