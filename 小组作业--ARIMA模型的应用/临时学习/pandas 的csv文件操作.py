import pandas
df = pandas.read_csv('hrdata.csv')
print(df)
print(df.head())  # head() 函数只会返回头5个数据
print(df.shape) # 可以返回一个元组（行数，列数）
print("=========================分割符===================================")
df2 = pandas.read_csv('hrdata.csv', index_col='Salary')
print(df2)
'''pandas库会自动识别CSV文件中的第一行包含列名，并且自动使用它们" 
与此同时他会自动使用从0开始的整数索引，那是因为我们没有告诉它我们的索引应该是什么
可以利用index_col 参数来指定索引的方式！
'''
print("================================分割符==================================")
df3 = pandas.read_csv('hrdata.csv', index_col='Name', parse_dates=['Hire Date'])
print(df3)
"可以利用parse_dates参数方法强制转换为日期格式输出"
print("=================================分割符========================")
"""
如果CSV文件的第一行中没有列名，则可以通过使用names可选参数来自己提供列名的列表
与此同时也可以利用这个来更改现在已有的列名，但是还必须使用header = 0 来告诉pandas
忽略现有的列名
"""
df4 = pandas.read_csv('hrdata.csv',
            index_col='Employee',
            parse_dates=['Hired'],
            header=0,
            names=['Employee', 'Hired','Salary', 'Sick Days'])
# 注意这里更改了列表名字
print(df4)
# 写入csv 文件
df4.to_csv("hrdata_modified.csv")

print("=================================分隔符===============================")

'''pandas的loc函数和iloc函数
loc指的是Selection by Label 函数
iloc指的是 Selection by Position即获取第几行第几列的数据 I指的是Ineger整数
'''
print(df3)
print()
print(df3.loc['Eric Idle3',"Hire Date"]) # 这里名字已经成为了标签
print()
print(df4.iloc[0,:]) # 全部都是数字，表示将第一行的数据取出
print()
print(df4.iloc[:3]['Salary']) # 表示取到前三行的工资内容