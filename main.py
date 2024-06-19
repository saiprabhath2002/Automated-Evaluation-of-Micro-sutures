import numpy as np 
import cv2
import os
import csv
import math
import sys
import shutil


def conv2gray(img):
    # print(img.shape)
    img=np.transpose(img,(2,0,1))
    # print(img.shape)
    gray=  np.uint8(0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])
    return(gray)

def resize(p,new_h=300):
    # print(p)
    img=cv2.imread(p)
    # print(img)
    # print("original size : ",img.shape,img.dtype)
    h,w=img.shape[:2]
    ratio=(w/h)
    new_w=int(new_h*ratio)
    resized_img=cv2.resize(img,(new_w,new_h))
    r_gray=conv2gray(resized_img)
    # r_gray=cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # print("after resize : ",r_gray.shape,r_gray.dtype)
    return(r_gray,resized_img)


def minfilter2(arr,h,w,flag):
    img=arr.copy()
    filter_img=[]
    for i in range(img.shape[0]-h+1):
        ro=[]
        for j in range(img.shape[1]-w+1):
            if(flag==0):
                ro.append(np.min(img[i:i+h,j:j+w]))
            else:
                ro.append(np.max(img[i:i+h,j:j+w]))
        filter_img.append(ro)
    filter_img=(np.array(filter_img))
    return (filter_img)


def invert(img):
    cp=img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(cp[i][j]==255):
                cp[i][j]=0
            else:
                cp[i][j]=255
    return cp


def gaussian_matrix(size, sigma):
    center = (size - 1) / 2
    x, y = np.meshgrid(np.arange(0, size), np.arange(0, size))
    exponent = -((x - center)**2 + (y - center)**2) / (2 * sigma**2)
    gaussian_values = np.exp(exponent) / (2 * np.pi * sigma**2)
    return gaussian_values / np.sum(gaussian_values)



def connected_component_acc_to_me( arr):
    h,w=arr.shape
    count = 0
    clus_size=[]
    stack=[]
    centroids=[]
    starting_pix=[]
    cpy=arr.copy()
    for i in range(arr.shape[0]-1,-1,-1):
        for j in range(arr.shape[1]):
            cluster_size=0
            ysum=0
            xsum=0
            all=[]
            if(cpy[i][j]==255):
                
                # cluster_size=1
                count+=1
                # starting_pix.append((j,h-i))
                stack.append((i,j))

                while(len(stack)!=0):
                    r,c=stack.pop()
                    ysum+=(h-r)
                    xsum+=c
                    cluster_size+=1
                    cpy[r,c]=0
                    all.append((c,h-r))
                    dir=[(r+1,c),(r-1,c),(r,c+1),(r,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1),(r-1,c-1)]  #,(r-1,c+1),(r+1,c-1),(r+1,c+1),(r-1,c-1)
                    for next in dir:
                        if(next[0]>=0 and next[0]<cpy.shape[0] and next[1]>=0 and next[1]<cpy.shape[1] and cpy[next[0],next[1]]==255):
                            stack.append((next[0],next[1]))
                duplicate=all
                center=(xsum//cluster_size,ysum//cluster_size)
                for poi in all:
                    if(poi[0]>center[0] and poi[1]>center[1]):
                        all.remove(poi)
                
                if(len(all)==0):
                    sorted_tuples=sorted(duplicate, key=lambda x: (x[0], x[1]))
                else:
                    sorted_tuples = sorted(all, key=lambda x: (x[0], x[1]))
                
                starting_pix.append(sorted_tuples[0])         
                centroids.append(center)
                clus_size.append(cluster_size)
    clus_size=np.array(clus_size)
    c=0
    for i in clus_size:
        # c+=round(i/np.mean(clus_size))
        if(i>100):
            c+=1
    return(c,starting_pix,centroids,clus_size)



def convolute(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Create padded image
    # padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')

    # Convolution result
    nh=image_height-kernel_height+1
    nw=image_width-kernel_width+1
    result = np.zeros((nh,nw))

    # Perform convolution
    for i in range(nh):
        for j in range(nw):
            # Perform element-wise multiplication and sum
            result[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
    # conv_img=Image.fromarray(result)
    return result  


def threshold_one_image(t,img):
    img_array=img.copy()
    # sample=np.array(Image.open(p))
    thresh_img_array=np.where(img_array>t,0,255)
    return(thresh_img_array)



def cal_varience(starting_pix,centroids,size,last_img):
    img_h,img_w=last_img.shape
    start=[]
    center=[]
    distance=[]
    angel=[]
    for i in range(len(starting_pix)):
        if(size[i]>100):
            # print(size[i],starting_pix[i],centroids[i])
            start.append(starting_pix[i])
            center.append(centroids[i])
    # print(center,start)
    for i in range(len(start)-1):
        distance.append(math.sqrt(((center[i][0]-center[i+1][0])/img_w)**2+((center[i][1]-center[i+1][1])/img_h)**2))
    for i in range(len(start)):
        # print(i)
        angel.append(math.degrees(math.atan((center[i][1]-start[i][1])/(center[i][0]-start[i][0]))))
    distance=np.array(distance)
    angel=np.array(angel)
    centroid_mean=np.mean(distance)
    angel_mean=np.mean(angel)
    centroid_varience=np.mean((distance-centroid_mean)**2)
    angel_varience=np.mean((angel-angel_mean)**2)
    # print(f'angel mean={angel_mean} varience={angel_varience} \ncentroid mean={centroid_mean} varience={centroid_varience}')
    # print(distance,angel)
    return(centroid_mean,centroid_varience,angel_mean,angel_varience)



def changed(p):
    l_img=[]
    # arr=resize(p)
    gray_img_array,resized_img=resize(p)
    g_conv=gaussian_matrix(3,3)
    # g_conv=(np.array([[1,2,1],
    #                 [2,4,2],
    #                 [1,2,1]]))/20
    # vertical_edge_conv=((np.array([[-1,-2,-1],
    #            [0,0,0],
    #            [1,2,1]]))/8)
    # horzontal_edge_conv=((np.array([[-1,0,1],
    #            [-2,0,2],
    #            [-1,0,1]]))/8)
    vertical_edge_conv=((np.array([[1,-2,-1],
                                    [0,0,0],
                                    [1,2,1]]))/8)
    horzontal_edge_conv=((np.array([[1,0,-1],   #sobelx
                                    [2,0,-2],
                                    [1,0,-1]]))/8)
    # gauss_img_array=nd.gaussian_filter(gray_img_array,sigma=1)
    # gauss_img=Image.fromarray(np.uint8(gauss_img_array))
    # l_img.append(Image.fromarray(gray_img_array))
    gauss_img_array=convolute(gray_img_array,g_conv)
    # plt.imshow(gauss_img_array,cmap='gray')
    # plt.show()
    # l_img.append(gauss_img)########
    v_edge_img_array=convolute(gauss_img_array,vertical_edge_conv)
    # plt.imshow(v_edge_img_array,cmap='gray')
    # plt.show()
    # l_img.append(v_edge_img)####
    h_edge_img_array=convolute(gauss_img_array,horzontal_edge_conv)
    # plt.imshow(h_edge_img_array,cmap='gray')
    # plt.show()
    # l_img.append(h_edge_img)##
    mag_final_edges_array=np.hypot(v_edge_img_array,h_edge_img_array)
    # plt.imshow(mag_final_edges_array,cmap='gray')
    # plt.show()
    
    thresh_img_array=threshold_one_image(30,mag_final_edges_array)
    # plt.imshow(thresh_img_array,cmap='gray')
    # plt.show()
    f3=minfilter2(thresh_img_array,3,3,1)
    # plt.imshow(f3,cmap='gray')
    # plt.show()
    f4=minfilter2(f3,2,6,0)
    # plt.imshow(f4,cmap='gray')
    # plt.show()
    f5=minfilter2(f4,3,1,1)
    # plt.imshow(f5,cmap='gray')
    # plt.show()
    f6=minfilter2(f5,1,10,0)
    # plt.imshow(f6,cmap='gray')
    # plt.show()
    f8=minfilter2(f6,1,2,0)
    # f7=invert(f8)
    # plt.imshow(f7,cmap='gray')
    # plt.show()
    findin=f8.copy()
    suture_c,start,centro,csize=connected_component_acc_to_me(findin)
    # suture_c,start,centro,csize=1,[],[],0
    return suture_c,start,centro,csize,resized_img,f8
        
        
    
def one_img_data(p,name_img,dir):    
    count,starting_pix,centroids,siz,resized_img,last_img=changed(p)
    hs,ws=resized_img.shape[:2]
    hl,wl=last_img.shape
    # print(f'satrt: {hs},{ws}')
    # print(f'satrt: {hl},{wl}')
    hdiff=hs-hl
    wdiff=ws-wl
    # print(f'diff : {hdiff},{wdiff}')
    print(name_img)
    print(f'suture count = {count}')
    resized_img=np.uint8(resized_img)
    centroid_mean,centroid_varience,angel_mean,angel_varience=cal_varience(starting_pix,centroids,siz,last_img)
    updated_centroid=[]
    
    for i in range(len(centroids)):
        if(siz[i]>100):
            # print(starting_pix[i],centroids[i],siz[i])
            updated_centroid.append(centroids[i])
    updated_centroid=np.array(updated_centroid)
    for i in range(len(updated_centroid)):
            updated_centroid[i][1]=hs-updated_centroid[i][1]
    for i in range(len(updated_centroid)):
         updated_centroid[i][0]+=(wdiff//2)
         updated_centroid[i][1]-=(hdiff//2)
    for i in range(len(updated_centroid)):
        cv2.circle(resized_img, (updated_centroid[i][0] , updated_centroid[i][1] ), 2, (0,0,255), -1)
    for i in range(len(updated_centroid) - 1):
        cv2.line(resized_img, updated_centroid[i], updated_centroid[i+1], 255, thickness=1)
 
    cv2.imwrite(dir+'/'+name_img,resized_img)
    # plt.imshow(resized_img,cmap='gray')
    # plt.show()
    answer=(name_img,count,centroid_mean,centroid_varience,angel_mean,angel_varience)
    return(answer)




def compare_img(p1,p2):
    sutur_count_img1,starting_pix_img1,centroids_img1,siz_img1,last_img1,drawon1=changed(p1)
    sutur_count_img2,starting_pix_img2,centroids_img2,siz_img2,last_img2,drawon2=changed(p2)
    print(f'img1 suture count = {sutur_count_img1}\nimg2 suture count = {sutur_count_img2}')
    print('for img1 : ')
    centroid_mean1,centroid_varience1,angel_mean1,angel_varience1=cal_varience(starting_pix_img1,centroids_img1,siz_img1,drawon1)
    print('for img2 : ')
    centroid_mean2,centroid_varience2,angel_mean2,angel_varience2=cal_varience(starting_pix_img2,centroids_img2,siz_img2,drawon2)
    print('based on inter ditance varience : ')
    if(centroid_varience1>centroid_varience2):
        print('img2')
        based_on_dist=2
    else:
        print('img1')
        based_on_dist=1
    print('based on angel varience : ')
    if(angel_varience1>angel_varience2):
        print('img2')
        based_on_angel=2
    else:
        print('img1')
        based_on_angel=1
    answer=(p1,p2,based_on_dist,based_on_angel)
    return answer


# print('hiii')

if(sys.argv[1]=="1"):
    dir_path=sys.argv[2]+'/'
    csvrows_all=[]
    new_directory='centroids_visualization'
    if os.path.exists(new_directory):
        shutil.rmtree(new_directory)
    os.makedirs(new_directory)
    files=os.listdir(dir_path)
    for i in files:
        csvrows_all.append(one_img_data(dir_path+i,i,new_directory))

    file_name=sys.argv[3]
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([ 'image_name', 'number of sutures',
    'mean inter suture spacing', 'variance of inter suture spacing', 'mean suture angle wrt x-axis', 'variance of suture angle wrt x-axis'])  # Writing header
        writer.writerows(csvrows_all) 

elif(sys.argv[1]=="2"):
    comparision_file_path=sys.argv[2]
    comaprision_data=[]
    with open(comparision_file_path,'r') as file:
        data=csv.reader(file)
        next(data)
        for i,r in enumerate(data):
            img1_path=str(r[0])
            img2_path=str(r[1])
            comaprision_data.append(compare_img(img1_path,img2_path))


    file_name=sys.argv[3]
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['img1_path', 'img2_path',
    'output_distance', 'output_angle'])  # Writing header
        writer.writerows(comaprision_data) 

    