import os
import pydicom
import numpy as np
from skimage import measure, morphology
from skimage.measure import regionprops
from scipy.ndimage import center_of_mass, label
# 图像处理模块导入measure（图像属性的测量，如相似性或等高线等），morphology（形态学操作，如开闭运算、骨架提取等）
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 定义加载CT扫描函数，载入一个扫描面，包含了多个切片(slices)，我们仅简化的将其存储为python列表，返回slices为3D张量
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s, force=True) for s in os.listdir(path) if s.endswith('.dcm')]  #读取dicom文件
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]), reverse=True)     #按z坐标对dicom进行排序
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness   # slicethickness为z方向切片厚度
        
    return slices

# 转换为ndarry；灰度值转换为HU单元
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])   #读取真实数据，将其存为image
    
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)   数据转换为int16型
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0    将CT扫面图之外的灰度值设为0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)  转化为HU值
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)                
            
        image[slice_number] += np.int16(intercept)      #由灰度值计算HU值（即CT值）
    
    return np.array(image, dtype=np.int16)

#肺部图像分割

#为了减少有问题的空间，我们可以分割肺部图像（有时候是附近的组织）。这包含一些步骤，包括区域增长和形态运算，此时，我们只分析相连组件。

#步骤如下：
#1.阈值图像（-320HU是个极佳的阈值，但是此方法中不是必要）
#2.处理相连的组件，以决定当前患者的空气的标签，以1填充这些二值图像
#3.可选：当前扫描的每个轴上的切片，选定最大固态连接的组织（当前患者的肉体和空气），并且其他的为0。以掩码的方式填充肺部结构。
#4.只保留最大的气袋（人类躯体内到处都有气袋

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)   #去除重复元素，排序

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

#定义
def segment_lung_mask(image, fill_lung_structures=True, dilate=False):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1    #选取大于-320的区域
    labels = measure.label(binary_image)                         #实现连通区域标记
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]     #设计背景标签
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2    #将患者周围充满空气
    
    # Method of filling the lung structures (that is superior to something like   填充肺组织
    # morphological closing)
    if fill_lung_structures==True:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):      #组合为索引序列，i为索引号，axial_slice为
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    
    if dilate==True:
        for i in range(binary_image.shape[0]):
            binary_image[i]=morphology.dilation(binary_image[i],np.ones([10,10]))
    return binary_image

# 标准化图像
def Standardizeimage(img):
    # function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    # 标准化像素值
    mean = np.mean(img)                #求所有元素均值
    std = np.std(img)                  #求所有元素标准差
    img = img-mean
    img = img/std

    middle = img[100:400,100:400]      #取图像中间300*300像素点
    mean = np.mean(middle)             #求所有元素均值
    max = np.max(img)                  #取元素最大值
    min = np.min(img)                  #取元素最小值
    # 为了改进阈值查找，这里移动了像素频谱上的下溢和溢出值
    img[img==max]=mean                 #将图像中最值元素变为均值
    img[img==min]=mean
    kmeans = KMeans(n_clusters=2).fit(middle.flatten().reshape(-1, 1))
    # shape为数组维度[a,b]，reshape为[a*b,1]，KMeans聚类算法聚合为两类
    del middle, mean, max, min

    centers = sorted(kmeans.cluster_centers_.flatten()) # 找出聚类中心，将中心列表化，获取已排序列表的副本为centers
    threshold = np.mean(centers)
    del centers, kmeans
    # print('Threshold: ', threshold)
    thresh_img = np.where(img<threshold,1.0,0.0)  # 阈值处理与二值化
    eroded = morphology.erosion(thresh_img,np.ones([4,4])) # 4x4矩阵腐蚀操作
    dilation = morphology.dilation(eroded,np.ones([10,10])) # 10x10矩阵膨胀操作
    del thresh_img, eroded

    labels = measure.label(dilation)
    del dilation

    regions = measure.regionprops(labels) # 返回所有连通区块的属性列表
    good_labels = []
    for prop in regions:
        B = prop.bbox                     # 得到边界外接框
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    del regions

    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    #
    # 这里的掩膜是肺部的掩膜--而不是节点
    # 只剩下肺部后，我们再做一次大的扩张
    # 为了填充肺部的掩膜 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    del labels, good_labels

    mask = morphology.dilation(mask,np.ones([10,10])) # 最后一次膨胀操作
    return mask*img

def nodule_coordinates(nodulelocations,meta):
    slices=nodulelocations["slice no."][nodulelocations.index[nodulelocations["case"]==int(meta["Patient Id"][-4:])]]
    xlocs=nodulelocations["x loc."][nodulelocations.index[nodulelocations["case"]==int(meta["Patient Id"][-4:])]]
    ylocs=nodulelocations["y loc."][nodulelocations.index[nodulelocations["case"]==int(meta["Patient Id"][-4:])]]
    nodulecoord=[]
    for i in range(len(slices)):
        nodulecoord.append([slices.values[i]-1,xlocs.values[i]-1,ylocs.values[i]-1])
    return nodulecoord

def plot_3d(image, threshold=-300):
    if image.shape[0] < image.shape[2]:
        image = np.transpose(image, (2, 1, 0))
    verts, faces, normals, vals = measure.marching_cubes(image, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])

    print("Ploting 3D image...\n")

    plt.show()

# 标准化图像
def Standardizeimage_3D(img):
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    # middle = img[100:400, 100:400, img.shape[2]-10:img.shape[2]+10]  # 取中间的300*300*300的像素
    middle = img[100:400, 100:400, : ]  # 取中间的300*300*img.shape[2]的像素
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)

    # 最值均值化
    img[img == max] = mean
    img[img == min] = mean

    # K-means 算法聚类,划分肺实质与其它物质的边界
    middle = middle.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=2).fit(middle)
    del middle, mean, max, min
    centers = sorted(kmeans.cluster_centers_.flatten())
    del kmeans
    threshold = np.mean(centers)
    del centers
    thresh_img = np.where(img < threshold, 1.0, 0.0)

    # Morphological operations
    eroded = morphology.erosion(thresh_img, np.ones([4, 4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10, 10]))
    del thresh_img, eroded

    # Label connected regions
    labels = measure.label(dilation)
    # label_vals = np.unique(labels)
    del dilation

    # Filter out small regions
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[4] - B[0] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    regions = []

    # Create lung mask
    mask = np.ndarray(img.shape, dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    del labels, good_labels
    
    # Perform final dilation
    mask = morphology.dilation(mask, np.ones([10, 10, 10]))
    
    return mask * img

# 从文件中处理图像
def process_image_from_file(ppix):
    processpix = np.ndarray([ppix.shape[0], 1, 512, 512], dtype=np.float32)
    for i in range(ppix.shape[0]):
        processpix[i, 0] = Standardizeimage(ppix[i])
    return processpix

# 获取最大结节的属性
def get_largest_nodule_properties(mask):
    mask[mask > 0.5] = 1
    mask[mask < 0.5] = 0
    mask = mask.astype(np.int8)
    labeled_array, nf = label(mask)
    areas_in_slice = []
    if nf > 1:
        for n in range(nf):
            lab = np.array(labeled_array)
            lab[lab != (n + 1)] = 0
            lab[lab == (n + 1)] = 1
            areas_in_slice.append(np.sum(lab))
        nlargest = areas_in_slice.index(max(areas_in_slice))
        labeled_array[labeled_array != (nlargest + 1)] = 0
        nodule_props = regionprops(labeled_array)
    else:
        nodule_props = regionprops(mask)
    area = nodule_props[0].area
    eccentricity = nodule_props[0].eccentricity
    diam = nodule_props[0].equivalent_diameter
    diammajor = nodule_props[0].major_axis_length
    spiculation = nodule_props[0].solidity
    return area, eccentricity, diam, diammajor, spiculation

def processimagenomask(img):
    # 标准化图像
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    middle = img[100:400, 100:400]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)

    # 根据上下限进行截断
    img[img == max] = mean
    img[img == min] = mean
    return img


def largestnodulecoordinates(mask):
    mask[mask > 0.5] = 1
    mask[mask < 0.5] = 0
    labeled_array, nf = label(mask)
    areasinslice = []
    if nf > 1:
        for n in range(nf):
            lab = np.array(labeled_array)
            lab[lab != (n + 1)] = 0
            lab[lab == (n + 1)] = 1
            areasinslice.append(np.sum(lab))
        nlargest = areasinslice.index(max(areasinslice))
        labeled_array[labeled_array != (nlargest + 1)] = 0
        com = center_of_mass(labeled_array)
    else:
        com = center_of_mass(mask)
    return [int(com[0]), int(com[1])]


def largestnodulearea(mask, table, i):
    mask[mask > 0.5] = 1
    mask[mask < 0.5] = 0
    labeled_array, nf = label(mask)
    areasinslice = []
    if nf > 1:
        for n in range(nf):
            lab = np.array(labeled_array)
            lab[lab != (n + 1)] = 0
            lab[lab == (n + 1)] = 1
            areasinslice.append(np.sum(lab))

        return max(areasinslice)
    else:
        return table["Area"][i]


def crop_nodule(coord, image):
    dim = 32
    return image[coord[0] - dim:coord[0] + dim, coord[1] - dim:coord[1] + dim]
    # 输出：带有恶性标签的结节的64x64图像
