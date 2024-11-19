import tkinter as ctk
import customtkinter as tk
tk.set_appearance_mode("Dark")
l=["",""]
def main():
    m=tk.CTk()
    m.title('Fake News Detection')
    m.geometry("800x500")
        
    def fp():
        for w in m.winfo_children():
            w.destroy()
        
        f = tk.CTkLabel(m,
                  text = "Fake " , font=("Helvetica", 60)).place(x = 40,
                                           y = 60)
        f1 = tk.CTkLabel(m,
                  text = "News Detection" , font=("Helvetica", 60)).place(x = 40,
                                           y = 150)
        btn = tk.CTkButton(m, text='Next', width=180,height=35 , command =sp)
        btn.place(x=600, y=450)
        m.mainloop()
        

    def sp():
        for w in m.winfo_children():
            w.destroy()
        def my():
            tp(1)
        def my2():
            tp(0)
        btn1 = tk.CTkButton(m, text='select datasets',font=("Helvetica", 15), width=300 ,height =60, command =my)
        btn2 = tk.CTkButton(m, text='continue ',font=("Helvetica", 15), width=300 , height =60,command =my2)
        btn1.place(x=260, y=150)
        btn2.place(x=260, y=300)
        btn = tk.CTkButton(m, text='Back', width=180 ,height=35 , command =fp)
        btn.place(x=25, y=450)
        m.mainloop()
    
    def tp(f):
        
        #print(l[0],l[1])
        if f==0:
            import pickle
            for w in m.winfo_children():
                w.destroy()
            model = pickle.load(open('model', 'rb'))
            def predict_cls():
                inp = T.get("1.0", "end-1c")
        #inp = inp.lower()
        #inp = punctuation_removal(inp)


                if inp is not None:
                    txt = [inp]
                p=model.predict(txt)
                if p==1:
                    s="the news is predicted as FAKE!!"
                else:
                    s="the news is predicted as REAL!!"
                f2=tk.CTkLabel(m,
                  text = s , font=("Helvetica", 20)).place(x = 250,
                                           y =400 )
                m.mainloop()

            
            
            f = tk.CTkLabel(m,
                  text = "Enter the News here to Test " , font=("Helvetica", 20)).place(x = 40,
                                           y = 30)
            T = tk.CTkTextbox(m, height = 300, width = 760)
            T.place(x=20,y=70)
            btn = tk.CTkButton(m, text='Test', width=180,height=35 ,command= predict_cls)
            btn.place(x=600, y=450)
            btn1 = tk.CTkButton(m, text='Back', width=180,height=35 , command =sp)
            btn1.place(x=25, y=450)
        
            m.mainloop()
            
            
            
           # a='C:\\Users\\shashank\\OneDrive\\Desktop\\fake_news_detection\\dataset\\True.csv'
           # b='C:\\Users\\shashank\\OneDrive\\Desktop\\fake_news_detection\\dataset\\Fake.csv'
           # print(a,b)
          #  frp(a,b)
        else:
            
            for w in m.winfo_children():
                w.destroy()
            from tkinter import filedialog
            
            def td():
                
                filename1=filedialog.askopenfilename()
                l[0]=filename1
                #print(filename1)
                #print(type(filename1))
                #file1=filename1
                

            def fd():
                
                filename2=filedialog.askopenfilename()
                l[1]=filename2
                print(filename2)
               # l[1]=filename2
               # print(type(filename2))
                #file2=filename2
                
            btn1 = tk.CTkButton(m, text='select True dataset',font=("Helvetica", 15), width=300 ,height =60,command=td)
            btn2 = tk.CTkButton(m, text='select False dataset',font=("Helvetica", 15), width=300 , height =60,command=fd)
            btn1.place(x=260, y=150)
            btn2.place(x=260, y=300)
            
            def check(f1,f2):
               
                
                #print(f1,f2)
                if f1==1 and f2==1:
                    frp(l[0],l[1])
                else:
                    st1=""
                    st2=""
                    if f1==0:
                        st1="please select a true dataset"
                    else:
                        st1=l[0]
                        

                    l1 = tk.CTkLabel(m,
                          text = st1 , font=("Helvetica", 20),text_color='red').place(x = 290,
                                          y = 215)
                        
                    if f2==0:
                        st2="please select a fake dataset"
                    else:
                        st2= l[1]
                    l2 = tk.CTkLabel(m,
                          text =  st2, font=("Helvetica", 20),text_color='red').place(x = 290,
                                          y = 365)
            def c():
                fg1=0
                fg2=0

                if l[0]!="":
                    fg1=1
                if l[1]!="":
                    fg2=1
                print(fg1,fg2)
                check(fg1,fg2)
            btn = tk.CTkButton(m, text='Next', width=180,height=35 , command =c)
            btn.place(x=600, y=450)
            btn4 = tk.CTkButton(m, text='Back', width=180,height=35 , command =sp)
            btn4.place(x=25, y=450)
            m.mainloop()

    


    def frp(filename1,filename2):
        for w in m.winfo_children():
                w.destroy()
    
        
        import pandas as pd
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        #import seaborn as sns 
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
        from sklearn import feature_extraction, linear_model, model_selection, preprocessing
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        #from google.colab import drive

        
    
        true=pd.read_csv(filename1)
        fake=pd.read_csv(filename2)

        fake['target'] = 1
        true['target'] = 0
        data = pd.concat([fake, true]).reset_index(drop = True)
        
       

        data.drop(["date"],axis=1 , inplace=True)
        from sklearn.utils import shuffle
        data = shuffle(data)
        data = data.reset_index(drop=True)
        data.drop(["title"],axis=1,inplace=True)

        data['text'] = data['text'].apply(lambda x: x.lower())

        import string

        def punctuation_removal(text):
            all_list = [char for char in text if char not in string.punctuation]
            clean_str = ''.join(all_list)
            return clean_str

        data['text'] = data['text'].apply(punctuation_removal)

            #nltk.download('stopwords')
            #from nltk.corpus import stopwords
            #stop = stopwords.words('english')

            #data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
      
        x_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)
        
        
        

        from xgboost import XGBClassifier

            #Create a xgboost Classifier

        clf = XGBClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42)
        pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', clf)])


        model = pipe.fit(x_train, y_train)
        prediction = model.predict(X_test)
        print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
        dct=dict()
        dct['XGBoost'] = round(accuracy_score(y_test, prediction)*100,2)
            
            #cm = metrics.confusion_matrix(y_test, prediction)
            #plot_confusion_matrix(cm, classes=['Fake', 'Real'])

        def predict_cls():
            inp = T.get("1.0", "end-1c")
            #inp = inp.lower()
            #inp = punctuation_removal(inp)


            if inp is not None:
                txt = [inp]
                p=model.predict(txt)
            if p==1:
                s="the news is predicted as FAKE!!"
            else:
                s="the news is predicted as REAL!!"
            f2=tk.CTkLabel(m,
                  text = s , font=("Helvetica", 20)).place(x = 250,
                                           y =400 )
            m.mainloop()

        f = tk.CTkLabel(m,
                  text = "Enter the News here to Test " , font=("Helvetica", 20)).place(x = 40,
                                           y = 30)
        T = tk.CTkTextbox(m, height = 300, width = 760)
        T.place(x=20,y=70)
        btn = tk.CTkButton(m, text='Test', width=180,height=35 ,command= predict_cls)
        btn.place(x=600, y=450)
        btn1 = tk.CTkButton(m, text='Back', width=180,height=35 , command =sp)
        btn1.place(x=25, y=450)
        
        m.mainloop()
        
            
        
            
        
    fp()
main()
    
