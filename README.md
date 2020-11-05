# Wheat_Detection
Proposing a FCN for semantic segmentation applied to wheat head detection. I used the learned the semantic segmentation to 
generate bounding boxes by applying standard blob detection techniques (openning, etc.).


This model is an implementation of an idea i had for the global wheat detection challenge on kaggle. The performance in itself is not great, the idea was to use semantic segmatation
because it was easier to learn and then create bounding boxes from there.


Here are examples of images, the learned segmentation and the bounding boxes generated.


![alt text](https://scontent-cdt1-1.xx.fbcdn.net/v/t1.15752-9/116588272_287267622595327_5322201165893978927_n.png?_nc_cat=101&ccb=2&_nc_sid=ae9488&_nc_ohc=ZDGc3RlOcMMAX-F0KH6&_nc_ht=scontent-cdt1-1.xx&oh=f035495abd37044e00c0d2bdecd9c83d&oe=5FC8AFFB)



![alt text](https://scontent-cdt1-1.xx.fbcdn.net/v/t1.15752-9/117242477_2715568455432531_3088788745022156913_n.png?_nc_cat=110&ccb=2&_nc_sid=ae9488&_nc_ohc=XfQKB4zfa0wAX-Fkpaa&_nc_ht=scontent-cdt1-1.xx&oh=1c07bcc7f4bf3442c3335da18af322f5&oe=5FCAD617)

![alt text](https://scontent-cdg2-1.xx.fbcdn.net/v/t1.15752-9/116604107_3204696599595505_6513750103228554475_n.png?_nc_cat=107&ccb=2&_nc_sid=ae9488&_nc_ohc=OIXE0c-LJK0AX9HoSId&_nc_ht=scontent-cdg2-1.xx&oh=b45843119accaf93ebd181a0149b22c7&oe=5FC813C5)


As we can see, the segmentation is pretty good, the bounding boxes, not so much.
