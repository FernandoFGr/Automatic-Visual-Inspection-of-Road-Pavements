import torch 
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from focalloss import FocalLoss
from torchvision import transforms, utils
import ignite
from myutils import transform, cal_iou, onehot, data_Train_transforms, data_Test_transforms
from PIL import Image
from dataset import tif2binary
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
try:
    from PIL import Image
except ImportError:
    import Image

def superposition_anotation(image,anotation):
    '''
    Input: crack image and anotation mask
    Output: superposition of the anotation mask on the image
    '''
    return torch.max(image, 255*anotation)
def matplotlib_imshow(img):
    '''
    Input: torchvision grid of images (torch tensor)
    - Plots the images
    '''
    #unnormalize
    npimg = img.numpy()
    plt.rcParams["figure.figsize"] = (50,15)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualize_example(data_loader=None, sample=None, model=None, classifier = None):
    '''
    Input: 
      - data_loader: torch DataLoader, which will generate the plotted image
      - sample: dictionary specifying {image, anotation} to be plotted
      - model: torch model to calculate predictions

    Output: image, groud truth, prediction, superposition of image and prediction
    '''
    if (data_loader is None) and (sample is None):
        return
    focallos = FocalLoss(gamma=2)
    if model is not None:
        pass
        #model.eval()
        #classifier.eval()
    if data_loader is not None:
        dataiter = iter(data_loader)
        image, anotation = dataiter.next()
    else:
        image, anotation = sample['image'], sample['anotation']

    image = image.to('cuda')
    anotation = anotation.type('torch.LongTensor')
    anotation = anotation.to('cuda')
    if model is not None:
        with torch.no_grad():
            outputs = model(image) #unsqueeze?
            outputs = classifier(outputs)
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                    transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                        std = [ 1., 1., 1. ]),
                                  ])

    image = invTrans(image[0])

    if model is not None:
        loss = focallos(outputs, anotation.long())
        #loss = criterion(outputs, torch.squeeze(anotation,1))
        #print(loss.item(), anotation.shape)

        preds = outputs.argmax(1) #predictions in black & white

        #F1 score:
        TP = torch.sum(preds[0]*anotation[0])
        FP = torch.sum(preds[0]*((anotation[0]+1)%2))
        FN = torch.sum(anotation[0]*((preds[0]+1)%2)) 
        print(TP, FP, FN)
        p = TP/(TP + FP + 1e-20)
        r = TP/(TP + FN + 1e-20)
        F1 = 2* (p * r)/(p + r + 1e-20)
        print("precision: ", p)
        print("recall: ", r)
        print("F1 score: ", F1)



        #preds = outputs[:,1,:,:] > 0.3
        preds_prob = outputs[:,1,:,:] #predictions in probability
        #print(outputs.shape, outputs.argmax(1).shape, "preds shape", preds.shape)
        preds = preds.to('cpu')
        preds = preds.type('torch.FloatTensor')
        preds_PIL=transforms.ToPILImage()(preds[0])
        preds_PIL = preds_PIL.convert('RGB')
        preds_prob = preds_prob.to('cpu')
        preds_prob = preds_prob.type('torch.FloatTensor')
        preds_prob_PIL=transforms.ToPILImage()(preds_prob[0])
        preds_prob_PIL = preds_prob_PIL.convert('RGB')
    anotation = anotation.to('cpu')
    anotation = anotation.type('torch.FloatTensor')
    #print(anotation[0].dtype)
    anotation_PIL=transforms.ToPILImage()(anotation[0])
    anotation_PIL = anotation_PIL.convert('RGB')
    image = image.to('cpu')
    image_PIL = transforms.ToPILImage()(image)
    image_PIL = image_PIL.convert('RGB')

    if model is not None:
        superposition = superposition_anotation(transforms.ToTensor()(image_PIL),
                                                transforms.ToTensor()(preds_PIL)) #replace 2 channels to 0s
        img_grid = torchvision.utils.make_grid([transforms.ToTensor()(image_PIL),
                                                transforms.ToTensor()(anotation_PIL),
                                                transforms.ToTensor()(preds_prob_PIL),
                                                superposition])
    else:
        superposition = superposition_anotation(transforms.ToTensor()(image_PIL),
                                                transforms.ToTensor()(anotation_PIL))
        img_grid = torchvision.utils.make_grid([transforms.ToTensor()(image_PIL),
                                                transforms.ToTensor()(anotation_PIL),
                                                superposition])

    # show images
    matplotlib_imshow(img_grid)
    #return img_grid



def visualize_real_and_sealed_cracks_superposition(data_loader, model, RC_classifier, SC_classifier):
    '''
    Input: Sealed crack train loader and model
    Saves a superposition of the real cracks and
    sealed cracks predictions in the folder "Superpositions"
    '''
    dataiter = iter(data_loader)
    num_images_to_create_superposition = 20
    for i in range(num_images_to_create_superposition):
        image, anotation = dataiter.next()
        image = image.to('cuda')
        anotation = anotation.type('torch.LongTensor')
        anotation = anotation.to('cuda')
        with torch.no_grad():
            encoder = model(image) #unsqueeze?
            RC_outputs = RC_classifier(encoder)
            SC_outputs = SC_classifier(encoder)

        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                        transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                            std = [ 1., 1., 1. ]),
                                      ])

        image = invTrans(image[0])

        foreground = (torch.max(RC_outputs[0,1], SC_outputs[0,1]) > 0.5)
        RC_pred = (foreground * (RC_outputs[0,1] >= SC_outputs[0,1])).to('cpu').type('torch.FloatTensor')
        SC_pred = (foreground * (RC_outputs[0,1] < SC_outputs[0,1])).to('cpu').type('torch.FloatTensor')
        output = torch.zeros(3, foreground.shape[0], foreground.shape[1]).to('cpu')
        foreground = foreground.to('cpu')
        output[0,:,:] = RC_pred
        output[1,:,:] = SC_pred

        preds = RC_outputs.argmax(1) #RC
        preds = preds.to('cpu')
        preds = preds.type('torch.FloatTensor')
        preds_PIL=transforms.ToPILImage()(preds[0])
        preds_PIL = preds_PIL.convert('RGB')



        anotation = anotation.to('cpu')
        anotation = anotation.type('torch.FloatTensor')
        anotation_PIL=transforms.ToPILImage()(anotation[0])
        anotation_PIL = anotation_PIL.convert('RGB')
        image = image.to('cpu')
        image_PIL = transforms.ToPILImage()(image)
        image_PIL = image_PIL.convert('RGB')

        plot_img = transforms.ToTensor()(image_PIL)
        plot_annot = transforms.ToTensor()(anotation_PIL)
        superposition = (plot_img * ((foreground + 1)%2)) + output 
        img_grid = torchvision.utils.make_grid([plot_img,
                                                #plot_annot,
                                                #output,
                                                #transforms.ToTensor()(preds_PIL),
                                                superposition])
        x = transforms.ToPILImage()(img_grid)
        x.save("superpositions/RCtest" + str(i)+ ".png")
    #return img_grid


def create_grid(img_path):
    # Open image file
    image = Image.open(img_path, grid_path)
    my_dpi=300.

    # Set up figure
    fig=plt.figure(figsize=(float(image.size[0])/my_dpi,float(image.size[1])/my_dpi),dpi=my_dpi)
    ax=fig.add_subplot(111)

    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    # Set the gridding interval: here we use the major tick interval
    myInterval=512.
    loc = plticker.MultipleLocator(base=myInterval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-', color='r')

    # Add the image
    ax.imshow(image)

    # Find number of gridsquares in x and y direction
    nx=abs(int(float(ax.get_xlim()[1]-ax.get_xlim()[0])/float(myInterval)))
    ny=abs(int(float(ax.get_ylim()[1]-ax.get_ylim()[0])/float(myInterval)))
    fig.savefig(grid_path)