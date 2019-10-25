# -*- coding: utf-8 -*-
import numpy as np
import random 
import loadnpz_and_plot as lnp
import h5py 
import matplotlib.pyplot as plt

# test1 settings
# select same number of sight lines from each column density bin
# use the total box for now

losdir = '/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/'

def generate_test1():
    #### actual test1
    #data = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0']
    #maxcol = np.max(data)
    #mincol = 12.
    #numbins = 10
    #sightlines_per_bin = 5
    #numpix = 32000  

    # [ 12.          12.55920906  13.11841812  13.67762718  14.23683624
    #  14.7960453   15.35525436  15.91446342  16.47367249  17.03288155
    #  17.59209061]

    ## test4
    #data = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_hydrogen_L0025N0376_27_test3.1_PtAb_C2Sm_8000pix_25.0slice_z-projection_wiEOS.npz')['arr_0']
    #maxcol = np.max(data)
    #mincol = np.min(data)
    #numbins = 10
    #sightlines_per_bin = 8
    #numpix = 8000

    ## test5 
    data = lnp.imreduce(np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_hydrogen_L0100N1504_27_test3.1_PtAb_C2Sm_28800pix_3.125slice_zcen-sum_x80.0-pm10.0_y5.0-pm10.0_z-projection_wiEOS_totalbox.npz')['arr_0'],9)
    maxcol = np.max(data)
    mincol = np.min(data)
    numbins = 10
    sightlines_per_bin = 20
    numpix = 3200

    bins = np.arange(mincol,1.001*maxcol,(1.0001*maxcol-mincol)/float(numbins))

    bininds = np.digitize(data,bins)
    indslist = [np.array(np.where(bininds == i+1)).T for i in range(numbins)]
    del bininds
    chosen = []
    for i in range(numbins):
        if len(indslist[i]) <= sightlines_per_bin:
            chosen += list(indslist[i])
        else:
            chosen += list(random.sample(indslist[i],sightlines_per_bin))

    fil = open(losdir + 'los_test6_L0100N1504.txt','w')
    fil.write('%i\n'%(len(chosen)))
    for i in range(len(chosen)):
        fil.write('%f\t%f\t0\n'%((chosen[i][0]+0.5)/(100./10.*float(numpix)) + 75./100., (chosen[i][1]+0.5)/(100./10.*float(numpix)))) 
    fil.close()

#test2 settings:
# lines around a column density peak: pixel centres, edges, corners
# check enviroments of 5 highest bin environments from test1: 41x41 pixels centered on value (in order of los file)
# first: central, highest-column values seem to be from unrresolved SPH particles
#        values can vary y ~0.44 dex between neighouring pixels
#        surroundings minumum in high 15s 
# 2nd:   lower-coldens surroundings minimum just below 14, even sharper contrasts bewteen neighbours (up to >1 dex). central stuff unresolved
# 3rd:   minimum in low 15s, less similar high-coldens surroundings, contrasts between 1 and 2
# 4th:   minimum in mid-15s, contrasts and appearance like 2nd
# 5th:   minumum below 15, very isolated and high contrasts
# use surroundings of 1 and 2: 5x5 projected pixels -> 11x11 centre, edge, and corner sightlines 

     
def generate_test2_test3():
    linecen1 =  [13066,   856]
    linecen2 = [  719, 23612]
    sightlines_per_length = 11
    pixel_per_length = 5
    numpix = 32000  

    lines_x = (linecen1[0] + 0.5*np.arange(-1*(sightlines_per_length/2),(sightlines_per_length+1)/2,1) + 0.5)/float(numpix)
    lines_y = (linecen1[1] + 0.5*np.arange(-1*(sightlines_per_length/2),(sightlines_per_length+1)/2,1) + 0.5)/float(numpix) 
    lines = np.empty((sightlines_per_length,sightlines_per_length,2))
    lines[:,:,0] = lines_x
    lines[:,:,1] = np.expand_dims(lines_y,1)
    
    fil = open(losdir + 'los_test2_pixind45_from_test1_and_surrounding.txt','w')
    fil.write('%i\n'%(lines.shape[0]*lines.shape[1]))
    for i in range(lines.shape[0]*lines.shape[1]):
        fil.write('%f\t%f\t0\n'%(lines[i/sightlines_per_length,i%sightlines_per_length,0], lines[i/sightlines_per_length,i%sightlines_per_length,1])) 
    fil.close()
    
    lines_x = (linecen2[0] + 0.5*np.arange(-1*(sightlines_per_length/2),(sightlines_per_length+1)/2,1) + 0.5)/float(numpix)
    lines_y = (linecen2[1] + 0.5*np.arange(-1*(sightlines_per_length/2),(sightlines_per_length+1)/2,1) + 0.5)/float(numpix) 
    lines = np.empty((sightlines_per_length,sightlines_per_length,2))
    lines[:,:,0] = lines_x
    lines[:,:,1] = np.expand_dims(lines_y,1)
    
    fil = open(losdir + 'los_test3_pixind46_from_test1_and_surrounding.txt','w')
    fil.write('%i\n'%(lines.shape[0]*lines.shape[1]))
    for i in range(lines.shape[0]*lines.shape[1]):
        fil.write('%f\t%f\t0\n'%(lines[i/sightlines_per_length,i%sightlines_per_length,0], lines[i/sightlines_per_length,i%sightlines_per_length,1])) 
    fil.close()


def generate_test7(data1 = None, data2 = None):
    #### actual test1
    #data = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0']
    #maxcol = np.max(data)
    #mincol = 12.
    #numbins = 10
    #sightlines_per_bin = 5
    #numpix = 32000  

    # [ 12.          12.55920906  13.11841812  13.67762718  14.23683624
    #  14.7960453   15.35525436  15.91446342  16.47367249  17.03288155
    #  17.59209061]

    ## test4
    #data = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_hydrogen_L0025N0376_27_test3.1_PtAb_C2Sm_8000pix_25.0slice_z-projection_wiEOS.npz')['arr_0']
    #maxcol = np.max(data)
    #mincol = np.min(data)
    #numbins = 10
    #sightlines_per_bin = 8
    #numpix = 8000

    ## test7: O7 and O8 peaks focussed
    if data1 is None:
        data1 = lnp.imreduce(np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_28800pix_3.125slice_zcen-sum_x80.0-pm10.0_y5.0-pm10.0_z-projection_T4EOS_totalbox.npz')['arr_0'],9)
    if data2 is None:
        data2 = lnp.imreduce(np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_28800pix_3.125slice_zcen-sum_x80.0-pm10.0_y5.0-pm10.0_z-projection_T4EOS_totalbox.npz')['arr_0'],9)
    maxcol1 = np.max(data1)
    mincol1 = np.min(data1)
    maxcol2 = np.max(data2)
    mincol2 = np.min(data2)
    numbins = 10
    sightlines_per_bin = 10
    numpix = 3200

    bins1 = np.arange(mincol1,1.001*maxcol1,(1.0001*maxcol1-mincol1)/float(numbins))
    bins2 = np.arange(mincol2,1.001*maxcol2,(1.0001*maxcol2-mincol2)/float(numbins))

    bininds1 = np.digitize(data1,bins1)
    indslist1 = [np.array(np.where(bininds1 == i+1)).T for i in range(numbins)]
    del bininds1
    bininds2 = np.digitize(data2,bins2)
    indslist2 = [np.array(np.where(bininds2 == i+1)).T for i in range(numbins)]
    del bininds2

    chosen = []
    for i in range(numbins):
        if len(indslist1[i]) + len(indslist2[i]) <= 2*sightlines_per_bin:
            sub1 = list(indslist1[i])
            sub2 = list(indslist2[i])
            chosen += list(set(sub1)+set(sub2))
        elif len(indslist1[i]) <= sightlines_per_bin:
            sub1= list(indslist1[i])
            sub2= list(random.sample(set(indslist2[i])-set(sub1),sightlines_per_bin))
            chosen += (sub1+sub2)
        elif len(indslist2[i]) <= sightlines_per_bin:
            sub1 = list(indslist2[i])
            sub2= list(random.sample(set(indslist1[i])-set(sub1),sightlines_per_bin))
            chosen += (sub1+sub2)
        else:
            bucket1 = indslist1[i]
            bucket2 = indslist2[i]
            sub = set()
            sublen = 0
            numsightlines1 = sightlines_per_bin
            numsightlines2 = sightlines_per_bin

            while sublen < 2*sightlines_per_bin:
                bucket1 = [tuple(xy) for xy in bucket1]
                bucket2 = [tuple(xy) for xy in bucket2]
                sub1 = list(random.sample(bucket1,numsightlines1))
                sub2 = list(random.sample(bucket2,numsightlines2))
                
                #print(str(sub1)+'\n'+str(sub2))
                sub = sub|(set(sub1)|set(sub2))
                sublen = len(sub)
                bucket1 = list(set(bucket1)-set(sub))
                bucket2 = list(set(bucket2)-set(sub))
                
                halfoverlap = (len(sub) - len(sub1) - len(sub2))/2 
                rand01 = np.random.randint(2)
                numsightlines1 -= (len(sub1) - halfoverlap - rand01)
                numsightlines2 -= (len(sub2) - halfoverlap - (1-rand01))
                chosen += list(sub)
    print('o7: %s\no8: %s'%(str([len(ind) for ind in indslist1]),str([len(ind) for ind in indslist2])))
    #print(str(data1[chosen]))
    #print(str(data2[chosen]))
    print('Got %i sightlines'%len(chosen))
    chosen = np.array(chosen)
    for fi in range(4):
        if len(chosen)%4 ==0:
            sel = slice(fi*len(chosen)/4, (fi+1)*len(chosen)/4,None)
        else:
            sel = slice(fi*(len(chosen)/4+1), (fi+1)*(len(chosen)/4+1),None)
        fil = open(losdir + 'los_test7_L0100N1504_q%i.txt'%(fi+1),'w')
        fil.write('%i\n'%(len(chosen)))
        for i in range(len(chosen)):
            fil.write('%f\t%f\t0\n'%((chosen[i][0]+0.5)/(100./10.*float(numpix)) + 75./100., (chosen[i][1]+0.5)/(100./10.*float(numpix)))) 
        fil.close()
    

def generate_test9(data1 = None, data2 = None):
    # evenly spaced bins in two ions

    ## test9: O7 and O8 peaks focussed
    #if data1 is None:
    #    data1 = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0'][75*320:85*320,0*320:10*320]
    #if data2 is None:
    #    data2 = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0'][75*320:85*320,0*320:10*320]
    ## test10: O7 and O8 peaks focussed; test 9 but for more sightlines and region 2 
    if data1 is None:
        data1 = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0'][10*320:20*320,75*320:85*320]
    if data2 is None:
        data2 = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0'][10*320:20*320,75*320:85*320]

    maxcol1 = np.max(data1)
    mincol1 = np.min(data1)
    maxcol2 = np.max(data2)
    mincol2 = np.min(data2)
    numbins = 16
    sightlines_per_bin = 16
    numpix = 3200

    bins1 = np.arange(mincol1,1.001*maxcol1,(1.0001*maxcol1-mincol1)/float(numbins))
    bins2 = np.arange(mincol2,1.001*maxcol2,(1.0001*maxcol2-mincol2)/float(numbins))

    bininds1 = np.digitize(data1,bins1)
    indslist1 = [np.array(np.where(bininds1 == i+1)).T for i in range(numbins)]
    del bininds1
    bininds2 = np.digitize(data2,bins2)
    indslist2 = [np.array(np.where(bininds2 == i+1)).T for i in range(numbins)]
    del bininds2

    chosen = []
    for i in range(numbins): # use sets to avoid sightline duplicates
        # need tuples at second level; sets do not work with arrays at this level
        sub1 = [tuple(xy) for xy in indslist1[i]]
        sub2 = [tuple(xy) for xy in indslist2[i]]
        # bin cannot be filled in both ions -> take all values in range for both
        if len(set(sub1) | set(sub2)) <= 2*sightlines_per_bin: 
            chosen += list(set(sub1) | set(sub2))
            print('%i sightlines added'%len(set(sub1)|set(sub2)))
        # one ion does not fill the bin -> take all values; subselection needed for the other
        # do subselection on not already selected sightlines
        elif len(sub1) <= sightlines_per_bin: 
            sub2= list(random.sample(set(sub2)-set(sub1),sightlines_per_bin))
            chosen += (sub1 | sub2)
            print('%i sightlines added'%len(set(sub1)|set(sub2)))
        elif len(sub2) <= sightlines_per_bin:
            sub1= list(random.sample(set(sub1)-set(sub2),sightlines_per_bin))
            chosen += (sub1 | sub2)
            print('%i sightlines added'%len(set(sub1)|set(sub2)))
        # need subselection for both; since ions may produce the same sightlines, iterate 
        # until the bin is filled with 2*sightlines_per_bin unique sightlines 
        else:
            bucket1 = sub1
            bucket2 = sub2
            sub = set()
            sublen = 0
            numsightlines1 = sightlines_per_bin
            numsightlines2 = sightlines_per_bin
            print('\nbin %i'%i)
            while sublen < 2*sightlines_per_bin: 
                print('sample1: %i, sample2: %i, total sample: %i'%(len(sub1),len(sub2),len(set(sub1)|set(sub2))) )
                print('numsightlines1: %i, numsightlines2: %i'%(numsightlines1,numsightlines2))
                sub1 = list(random.sample(bucket1,numsightlines1))
                sub2 = list(random.sample(bucket2,numsightlines2))
                
                #print(str(sub1)+'\n'+str(sub2))
                sub = sub|(set(sub1)|set(sub2))
                sublen = len(sub)
                bucket1 = list(set(bucket1)-set(sub))
                bucket2 = list(set(bucket2)-set(sub))
                
                # divide any sightline overlap etween the two bin selections equally 
                # over both numbers of sightlines, dividing any odd remiander randomly
                overlap = -1*(len(sub) - len(sub2) - len(sub2)) 
                rand01 = np.random.randint(2)
                numsightlines1 -= (len(sub1) - overlap/2 - rand01*overlap%2)
                numsightlines2 -= (len(sub2) - overlap/2 - (1-rand01)*overlap%2)
                # one of the ion buckets could be taking away everything from the other 
                # we know that there are enough sightlines in total, so just reshuffle 
                # numsightlines; also applies when odd remainders
                if numsightlines1 > len(bucket1) or numsightlines2 < 0:
                    print('reshuffling 1 to 2')
                    numsightlines2 += numsightlines1 - len(sub1)
                    numsighlines1 = len(bucket1)
                if numsightlines2 > len(bucket2) or numsightlines1 < 0:
                    print('reshuffling 2 to 1')
                    numsightlines1 += numsightlines2 - len(sub2)
                    numsighlines2 = len(bucket2)

            chosen += list(sub)
            print('%i sightlines added'%len(sub))
    print('o7: %s\no8: %s'%(str([len(ind) for ind in indslist1]),str([len(ind) for ind in indslist2])))
    #print(str(data1[chosen]))
    #print(str(data2[chosen]))
    print('Got %i sightlines, %i distinct'%(len(chosen),len(set([tuple(xy) for xy in chosen]))))
    chosen = np.array(chosen)

    fil = open(losdir + 'los_test10_L0100N1504.txt','w')
    fil.write('%i\n'%(len(chosen)))
    for i in range(len(chosen)):
        fil.write('%f\t%f\t0\n'%((chosen[i][0]+0.5)/(10.*float(numpix)) + 10./100., (chosen[i][1]+0.5)/(10.*float(numpix))+75./100)) 
    fil.close()

    return [data1[chosen[i][0],chosen[i][1]] for i in range(len(chosen))],  [data2[chosen[i][0],chosen[i][1]] for i in range(len(chosen))]


def generate_sample1():
    # evenly spaced bins in O7
    ## col. dens. minimum is set so that the first bins are well within the linear region of the curve of growth
     

    data = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0']
    maxcol = np.max(data)
    mincol = 13.
    numbins = 128
    sightlines_per_bin = 128
    numpix = 32000
    remainder = 0 # if a bins has < sightlines_per_bin sightlines, add the remainder to the next bin(s)
    maxaddtobin = sightlines_per_bin # max number of remainder sightlines to add to a single bin (first round) 
    binswithroom = [] # list of bins to take sightlines from if there are slots over after the loop
    # (number of bins needs to be fixed to guarantee divisibility by the number of cores)
    if numbins*sightlines_per_bin > numpix**2:
        print('You cannot select %i sightlines from a sample of %i pixels.'%(numbins*sightlines_per_bin, numpix**2))
        return None

    bins = np.arange(mincol,1.001*maxcol,(1.0001*maxcol-mincol)/float(numbins)) # generate numbins+1 values -> numbins bin edges

    bininds = np.digitize(data,bins)
    indslist = [np.array(np.where(bininds == i+1)).T for i in range(numbins)]
    del bininds
    chosen = []

    for i in range(numbins-1,-1,-1): # start at last index: if there is a remainder, it is preferable to sample more from the large column density bins
        numsightlines = sightlines_per_bin

        if len(indslist[i]) <= sightlines_per_bin: # no need to select if there are <= sightlines_per_bin sightlines in the bin
            chosen += list(indslist[i])
            remainder += sightlines_per_bin - len(indslist[i])

        elif len(indslist[i]) - sightlines_per_bin > min(remainder,maxaddtobin): # put <= sightlines_per_bin of the left over sightline slots into this bin
            numsightlines += min(remainder,maxaddtobin)
            remainder -= min(remainder,maxaddtobin)
            chosensub = list(random.sample(indslist[i],numsightlines))
            chosen += chosensub
            binswithroom += [(i,chosensub)]
        else: # indslist[i] has more elements than sightlines_per_bin, less than max sightlines that can be selected -> no selection needed
            chosen += list(indslist[i])
            remainder -= (len(indslist[i]) - sightlines_per_bin) 
        if remainder < 0:
            print('Error: remainder < 0 ! at bin %i, indslist length %i'%(i, len(indslist[i])))
            return 0

    # check if all sightline slots have been used
    if remainder == 0 and len(chosen) == numsightlines*sightlines_per_bin:
        print('Sightline selection seems to be succesful.')
    elif remainder != numsightlines*sightlines_per_bin - len(chosen):
        print('Error: %i/%i sightlines selected, but remainder is %i'%(len(chosen),numsightlines*sightlines_per_bin,remainder))
        return None
    else: # if there are slots remaining, take more from all bins, starting at the highest column densities
        print('Need to select %i additional sightlines.')
        ind = 0 # for binswithroom: last bins come first there
        while remainder  > 0:   
            chosensub = list(set(indslist[binswithroom[ind][0]]) - set(binswithroom[ind][1]))
            if len(chosensub) > remainder: # need to make a selection
                chosen += list(random.sample(chosensub,remainder))
                remainder  = 0
            else:
                chosen += chosensub
                remainder -= len(chosensub)
                ind += 1
    # repeat check
    if remainder == 0 and len(chosen) == numsightlines*sightlines_per_bin:
        print('Sightline selection seems to be succesful.')
    elif remainder != numsightlines*sightlines_per_bin - len(chosen):
        print('Error: %i/%i sightlines selected, but remainder is %i'%(len(chosen),numsightlines*sightlines_per_bin,remainder))
        return None
    else:
        print('Unanticipated error: the mop-up assignment of sightlines seems to have gone wrong')
        print('%i/%i sightlines selected, remainder is %i'%(len(chosen),numsightlines*sightlines_per_bin,remainder))   
        return None
          
   
    fil = open(losdir + 'los_sample1_L0100N1504.txt','w')
    fil.write('%i\n'%(len(chosen)))
    for i in range(len(chosen)):
        fil.write('%f\t%f\t0\n'%((chosen[i][0]+0.5)/(float(numpix)), (chosen[i][1]+0.5)/(float(numpix)))) 
    fil.close()

    return [data[chosen[i][0],chosen[i][1]] for i in range(len(chosen))]





def generate_grid1():
    linesperside = 128
    lengthoffset = 0.5/32000. #put lines at pixel centres (only works in linesperside divides #pixels per side)

    # evenly spaced; offset should be smaller than the spacing for predictable behaviour
    xs = np.arange(linesperside)/float(linesperside) + lengthoffset
    ys = np.arange(linesperside)/float(linesperside) + lengthoffset
    
    fil = open(losdir + 'los_grid1_L0100N1504.txt','w')
    fil.write('%i\n'%(linesperside**2))
    for i in range(linesperside):
        for j in range(linesperside):
            fil.write('%f\t%f\t0\n'%(xs[i], ys[j])) 
    fil.close()
    
    #grid1b: checked run1 output (incomplete), and made a new file of all the positions not already covered 

def generate_missing(old_losfile,new_losfile,outputfile):
    # may not work: duplicated sightlines seem to have been used in grid1b
    import specwiz_proc as sp
    #if isinstance(outputfile,sp.Specout):
    #    pass
    #else:
    #    outputfile = sp.Specout(outputfile,getall=False) # only need positions
    
    # read in old/total grid positions to allpositions
    allpositions = []
    posfile_old = open(old_losfile,'r')
    for line in posfile_old:
        line = line[:-1] #remove '\n'
        xyz = line.split('\t') # list of the entries separated y tabs
        if len(xyz) == 1: #first line
            sightlines = int(xyz[0])
        else:
            x = float(xyz[0])
            y = float(xyz[1])
            allpositions.append(np.array([x,y]))
    allpositions = np.array(allpositions)
    posfile_old.close()

    # get fractional positions back from outputfile
    posout = outputfile.positions/(outputfile.slicelength*sp.hcosm)

    # find indices in allpositions that correspond to positions best matched y sightlines for which we already have output  
    # minumum distance used since floating point errors prohibit exact matching  
    indsclosest = ( (np.expand_dims(posout[:,0],1) - np.expand_dims(allpositions[:,0],0))**2 +\
                    (np.expand_dims(posout[:,1],1) - np.expand_dims(allpositions[:,1],0))**2 ).argmin(axis=1)
    # the other indices are what we want
    indstodo = set(range(len(allpositions))) - set(indsclosest)
    positionstodo = allpositions[list(indstodo)]
    numpostodo = len(positionstodo)
    print('Found %i sightlines not yet done for %i total, %i done.'%(numpostodo,len(allpositions),len(posout)))
    
    #newfile = open(new_losfile,'w')
    #newfile.write('%i\n'%numpostodo)
    #for i in range(numpostodo):
    #    newfile.write('%f\t%f\t0\n'%(positionstodo[i][0], positionstodo[i][1])) 
    #newfile.close()

    return np.min( (np.expand_dims(posout[:,0],1) - np.expand_dims(allpositions[:,0],0))**2 +\
                   (np.expand_dims(posout[:,1],1) - np.expand_dims(allpositions[:,1],0))**2, axis=1)


def generate_colbin_sample():
    # approx. evenly spaced bins in each data<i> quantity
    ## col. dens. minimum is set so that the first bins are well within the linear region of the curve of growth
     
    # sample for the z=0.1 O7/O8-selected specwizard run for EW from column density
    #data1 = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0']
    #data2 = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0']
    #name = 'los_sample2_o7-o8_L0100N1504.txt'

    #datas = [data1, data2]
    #maxcols = [np.max(data) for data in datas]
    #mincols = [13., 13.]
    #numbins = 128
    #sightlines_per_bin = 128
    #numpix = 32000 #used to convert index -> position
    
    #maxaddtobin = sightlines_per_bin # max number of remainder sightlines to add to a single bin (first round) 
    # (number of bins needs to be fixed to guarantee divisibility by the number of cores specwizard is run on)

    # sample for the z=0.1 O6/O7/O8-selected specwizard run for EW from column density
    data1 = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0']
    data2 = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz')['arr_0']
    data3 = np.load('/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz')['arr_0']
    name = 'los_sample2_o6-o7-o8_L0100N1504.txt'

    datas = [data1, data2, data3]
    maxcols = [np.max(data) for data in datas]
    mincols = [13., 13., 13.]
    numbins = 128
    sightlines_per_bin = 128
    numpix = 32000 #used to convert index -> position
    
    maxaddtobin = sightlines_per_bin # max number of remainder sightlines to add to a single bin (first round) 
    # (number of bins needs to be fixed to guarantee divisibility by the number of cores specwizard is run on)


    # below this line: everything should allow modification of the above, including number of data sets
    if numbins*sightlines_per_bin > numpix**2:
        print('You cannot select %i sightlines from a sample of %i pixels.'%(numbins*sightlines_per_bin, numpix**2))
        return None

    binss = [np.arange(mincols[i],1.001*maxcols[i],(1.0001*maxcols[i]-mincols[i])/float(numbins)) for i in range(len(datas))] # generate numbins+1 values -> numbins bin edges

    binindss = [np.digitize(datas[i],binss[i]) for i in range(len(datas))]
    indslist = [[np.array(np.where(binindss[dataind] == i+1)).T for i in range(numbins)] for dataind in range(len(datas))]
    indslist = [[ indslist[dataind][binind][:,0]*numpix + indslist[dataind][binind][:,1] for binind in range(numbins)] for dataind in range(len(datas))] #set cannot contain arrays -> change to tuples in innermost nest
    del binindss
    chosen = set()
    #bins_allchosen = [[] for dataind in range(len(datas))] # track where the sampling is complete for each data set
    

    sightlines_per_bin_per_ion = sightlines_per_bin/len(datas) #fill in any remainder from the overlap
    final_sample_size = numbins*sightlines_per_bin
    collected_sample_size = len(chosen)
    remainder = 0 # if a bins has < sightlines_per_bin sightlines, add the remainder to the next bin(s)
    sightlines_per_bin_per_ion = (sightlines_per_bin + min(remainder,maxaddtobin))/len(datas) # may have a remainder in integer division
    

    # outer loop: over bins (largest first, to get remainder mainly put there)
    for binind in range(numbins-1,-1,-1):
        print("Doing bin %i"%binind)
        target_collected_sample_size = (numbins - binind)*sightlines_per_bin # at the end, remainder = this - number collected 
        lenchosenstart = len(chosen)

        # check if the ions together can fill the bin
        sightlines_all_ions = set()
        for dataind  in range(len(datas)):
            sightlines_all_ions |= set(indslist[dataind][binind])

        if len(sightlines_all_ions) <= sightlines_per_bin + min(remainder,maxaddtobin):
            chosen |= sightlines_all_ions
            remainder = target_collected_sample_size - len(chosen) # we can't just subtract the number of sightlines added, since some might have already been selected in a previous bin on a different ion
            continue #this bin is filled

        # inner loop: over ions for each bin
        # cannot just take all sightlines: go ion by ion to ensure a sufficient sample size in all ions where possible
        # first, assign each ion its allotted amount from sightlines_per_bin and the remainder
       
        for dataind in range(len(datas)):

            # if the number of sightlines for the ion is small enough, just add them all
            if len(indslist[dataind][binind]) <= sightlines_per_bin_per_ion:
                chosen |= set(indslist[dataind][binind])
                sightlines_all_ions -= set(indslist[dataind][binind]) #remove from set of sightlines to consider in next steps
            # otherwise, take a random sample
            else:
                sample_temp = set(random.sample(indslist[dataind][binind],sightlines_per_bin_per_ion))
                chosen |= sample_temp
                sightlines_all_ions -= sample_temp
        
        # if the allotment for this bin has not been exceeded, and there is something left to select from, do the selection
        # loop is used since some of the new selection may overlap with previous selections 
        lefttodo = lenchosenstart + sightlines_per_bin + min(remainder,maxaddtobin) -len(chosen)      
        while(lefttodo > 0 and len(sightlines_all_ions) > 0):
            if lefttodo >= len(sightlines_all_ions):
                sample_temp = sightlines_all_ions
            else:     
                sample_temp = set(random.sample(sightlines_all_ions,lefttodo))
            chosen |= sample_temp
            sightlines_all_ions -= sample_temp
            lefttodo = lenchosenstart + sightlines_per_bin + min(remainder,maxaddtobin) -len(chosen)

        # set up remainder for the next loop
        remainder = target_collected_sample_size - len(chosen) 

    # if there is a remainder left, just add some random sightlines
    if remainder > 0:
        notchosen = set([indslist[0][binind] for binind in range(numbins)]) - chosen
        chosen |= set(random.sample(notchosen,remainder))
   
    fil = open(losdir + name,'w')
    fil.write('%i\n'%(len(chosen)))
    chosen = list(chosen)
    for i in range(len(chosen)):
        fil.write('%f\t%f\t0\n'%((chosen[i]/numpix+0.5)/(float(numpix)), (chosen[i]%numpix+0.5)/(float(numpix)))) 
    fil.close()

    dataselection = np.array([np.array([data[chosen[i]/numpix,chosen[i]%numpix] for i in range(len(chosen))]) for data in datas])
    binsamples = np.array([[len(indslist[dataind][binind]) for binind in range(numbins)] for dataind in range(len(datas))]) 
    np.savez(losdir + name + '_data', selectedvalues = dataselection, binsused = np.array(binss), originaldataperbin = binsamples)
    return dataselection, binss, binsamples 


def generate_lumped_sample():
    # sample for the z=0.1 O6/O7/O8-selected specwizard run for EW from column density (sample3)
    file1 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz'
    file2 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz'
    file3 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
    data1 = np.load(file1)['arr_0']
    data2 = np.load(file2)['arr_0']
    data3 = np.load(file3)['arr_0']
    name = 'los_sample3_o6-o7-o8_L0100N1504'

    datas = [data1, data2, data3]
    files = [file1, file2, file3]
    maxcols = [np.max(data) for data in datas]
    mincols = [13., 13., 13.]
    numbins = 128
    sightlines_per_bin = 128
    numpix = 32000 #used to convert index -> position
    
    maxaddtobin = sightlines_per_bin # max number of remainder sightlines to add to a single bin (first round) 
    # (number of bins needs to be fixed to guarantee divisibility by the number of cores specwizard is run on)


    # below this line: everything should allow modification of the above, including number of data sets
    numions = len(datas)
    if numbins*sightlines_per_bin > numpix**2:
        print('You cannot select %i sightlines from a sample of %i pixels.'%(numbins*sightlines_per_bin, numpix**2))
        return None
    binss = [np.arange(mincols[i],1.001*maxcols[i],(1.0001*maxcols[i]-mincols[i])/float(numbins)) for i in range(numions)] # generate numbins+1 edges values -> numbins bins
    # store metadata
    mdfile = h5py.File(losdir + name + '_data.hdf5', 'w')
    header = mdfile.create_group('Header')
    header.attrs.create('input_files', np.array(files))
    header.attrs.create('los_file', losdir + name + '.txt')
    header.attrs.create('numbins', numbins)
    header.attrs.create('total_sightlines_per_bin', sightlines_per_bin)
    header.attrs.create('min_values', mincols)
    header.attrs.create('max_values', maxcols)
    header.attrs.create('numpix', numpix)   
    header.create_dataset('bins', data = binss)

    binindss = [np.digitize(datas[i],binss[i]) for i in range(len(datas))]
    indslist = [[np.array(np.where(binindss[dataind] == i+1)).T for i in range(numbins)] for dataind in range(numions)]
    indslist = [[ indslist[dataind][binind][:,0]*numpix + indslist[dataind][binind][:,1] for binind in range(numbins)] for dataind in range(numions)] #set cannot contain arrays -> change to tuples in innermost nest; list of flattened indices corresponding in each bin for each ion
    del binindss
    chosen = set()
    chosens = [set() for data in datas]
    #bins_allchosen = [[] for dataind in range(len(datas))] # track where the sampling is complete for each data set
        
    sightlines_per_bin_per_ion = sightlines_per_bin/numions #any remainder here goed into the remainder accounting
    final_sample_size = numbins*sightlines_per_bin
    collected_sample_size = len(chosen)
    sightlines_per_ion = final_sample_size/numions 
    maxaddtobin = sightlines_per_bin_per_ion

    doselectionloop = True
    while(doselectionloop):
        # outer loop: over ions, independent of each other
        for dataind in range(numions):
            print("Doing file %i"%dataind)
            remainder_ion = sightlines_per_ion - numbins*sightlines_per_bin_per_ion 
            print("initial remainder: %i"%remainder_ion)
            for binind in range(numbins-1,-1,-1): 
                sample_bin = set(indslist[dataind][binind]) - chosens[dataind]  # sightlines in this bin that have not been selected yet for this ion
                targetnumsightlines_sub = sightlines_per_bin_per_ion + min(remainder_ion, maxaddtobin) # number of sightlines we want from this bin   
                if targetnumsightlines_sub == 0: # on non-first loops: remainder used up, no baseline number to select
                    print('Out of sightlines for file %i'%dataind)
                    break      
                # if the bin only has a small number of sightlines, use them all
                if len(sample_bin) <= targetnumsightlines_sub:
                    chosens[dataind] |= sample_bin
                    remainder_ion += sightlines_per_bin_per_ion - len(sample_bin) # negative difference if some of the new sightlines are coming out of the remainder
                # otherwise, take a random sample
                else:
                    sample_temp = set(random.sample(sample_bin,targetnumsightlines_sub))
                    chosens[dataind] |= sample_temp
                    remainder_ion += sightlines_per_bin_per_ion - targetnumsightlines_sub # negative difference if some of the new sightlines are coming out of the remainder
            print("final remainder: %i"%remainder_ion)
        # get total sample set, check size
        for sub in chosens: 
            chosen |= sub
        collected_sample_size = len(chosen)
        remainder_total = final_sample_size - collected_sample_size
        print("Total remainder: %i"%remainder_total)
        # as long as there is enough to divide over the ions uniformly, redo the bin loop
        if remainder_total > numions-1: # can still divide more of the sightlines
            sightlines_per_ion = remainder_total/numions
            sightlines_per_bin_per_ion = sightlines_per_ion/numbins # may be zero: in this case, just the remainder is divided up
            maxaddtobin = max(sightlines_per_bin_per_ion, 1) #if sightlines_per_bin_per_ion=0, still want to add to bins
        else: # end the while loop
            doselectionloop = False  

    # if there is a remainder left, just add some random sightlines
    if remainder_total > 0:
        notchosen = set([indslist[0][binind] for binind in range(numbins)]) - chosen
        chosen_remainder = random.sample(notchosen,remainder_total)
    else:
        chosen_remainder = set()

    chosens = [np.array(list(chosen_sub)) for chosen_sub in chosens]
    chosen = np.array(list(chosen))
    chosen_remainder = np.array(list(chosen_remainder))
    # save metadata
    selection = mdfile.create_group('Selection') 
    for i in range(numions):
        group = mdfile.create_group('Selection/file%i'%i)
        group.attrs.create('filename', files[i])
        group.create_dataset('bins',data = binss[i])
        group.create_dataset('selected_pixels_thision', data=np.array([chosens[i]/numpix,chosens[i]%numpix]).T)
        group.create_dataset('selected_values_thision', data=np.array([datas[i][ch/numpix,ch%numpix] for ch in chosens[i]]))
        group.create_dataset('selected_values_allions', data=np.array([datas[i][ch/numpix,ch%numpix] for ch in chosen]))
        group.create_dataset('sample_pixels_per_bin', data=np.array([len(indslist[i][binind]) for binind in range(numbins)]))    
    selection.create_dataset('selected_pixels_allions', data=np.array([chosen/numpix,chosen%numpix]).T)
    group = mdfile.create_group('Selection/remainder')
    group.create_dataset('selected_pixels', data=np.array([chosen_remainder/numpix,chosen_remainder%numpix]).T)  
    mdfile.close()
    
    # make los file for specwizard
    fil = open(losdir + name + '.txt','w')
    fil.write('%i\n'%(len(chosen)))
    chosen = list(chosen)
    for i in range(len(chosen)):
        fil.write('%f\t%f\t0\n'%((chosen[i]/numpix+0.5)/(float(numpix)), (chosen[i]%numpix+0.5)/(float(numpix)))) 
    fil.close()

    return None


def generate_lumped_sample_z1check(kind='check19'):
    if kind == 'check19':
        # sample for the z=0.1 O6/O7/O8-selected specwizard run for EW from column density (sample3)
        file1 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        file2 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        #file3 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        data1 = np.load(file1)['arr_0']
        data2 = np.load(file2)['arr_0']
        #data3 = np.load(file3)['arr_0']
        name = 'los_sample3-verf_o7-o8_L0100N1504_snap19'

        datas = [data1, data2]
        files = [file1, file2]
        maxcols = [np.max(data) for data in datas]
        mincols = [15.0, 15.5]
        numbins = 32
        sightlines_per_bin = 8
        numpix = 32000 #used to convert index -> position
    
        maxaddtobin = sightlines_per_bin # max number of remainder sightlines to add to a single bin (first round) 
        # (number of bins needs to be fixed to guarantee divisibility by the number of cores specwizard is run on)

    elif kind == 'sample4_19':
        # sample for the z=0.1 O6/O7/O8-selected specwizard run for EW from column density (sample3)
        file1 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        file2 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        #file3 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        data1 = np.load(file1)['arr_0']
        data2 = np.load(file2)['arr_0']
        #data3 = np.load(file3)['arr_0']
        name = 'los_sample4_o7-o8_L0100N1504_snap19'

        datas = [data1, data2]
        files = [file1, file2]
        maxcols = [np.max(data) for data in datas]
        mincols = [13.0, 13.0]
        numbins = 128
        sightlines_per_bin = 128
        numpix = 32000 #used to convert index -> position
    
    elif kind == 'sample5_26':
        # sample for the z=0.1 O6/O7/O8-selected specwizard run for EW from column density (sample3)
        file1 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        file2 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        #file3 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        data1 = np.load(file1)['arr_0']
        data2 = np.load(file2)['arr_0']
        #data3 = np.load(file3)['arr_0']
        name = 'los_sample5_o7-o8_L0100N1504_snap26'

        datas = [data1, data2]
        files = [file1, file2]
        maxcols = [np.max(data) for data in datas]
        mincols = [13.0, 13.0]
        numbins = 128
        sightlines_per_bin = 128
        numpix = 32000 #used to convert index -> position
        
    # below this line: everything should allow modification of the above, including number of data sets
    numions = len(datas)
    if numbins*sightlines_per_bin > numpix**2:
        raise RuntimeError('You cannot select %i sightlines from a sample of %i pixels.'%(numbins*sightlines_per_bin, numpix**2))
    binss = [np.arange(mincols[i], 1.001*maxcols[i], (1.0001 * maxcols[i] - mincols[i]) / float(numbins)) for i in range(numions)] # generate numbins+1 edges values -> numbins bins
    # store metadata
    mdfile = h5py.File(losdir + name + '_data.hdf5', 'w')
    header = mdfile.create_group('Header')
    header.attrs.create('input_files', np.array(files))
    header.attrs.create('los_file', losdir + name + '.txt')
    header.attrs.create('numbins', numbins)
    header.attrs.create('total_sightlines_per_bin', sightlines_per_bin)
    header.attrs.create('min_values', mincols)
    header.attrs.create('max_values', maxcols)
    header.attrs.create('numpix', numpix)   
    header.create_dataset('bins', data = binss)

    binindss = [np.digitize(datas[i], binss[i]) for i in range(len(datas))]
    indslist = [[np.array(np.where(binindss[dataind] == i+1)).T for i in range(numbins)] for dataind in range(numions)]
    indslist = [[ indslist[dataind][binind][:,0]*numpix + indslist[dataind][binind][:,1] for binind in range(numbins)] for dataind in range(numions)] #set cannot contain arrays -> change to tuples in innermost nest; list of flattened indices corresponding in each bin for each ion
    del binindss
    chosen = set()
    chosens = [set() for data in datas]
    #bins_allchosen = [[] for dataind in range(len(datas))] # track where the sampling is complete for each data set
        
    sightlines_per_bin_per_ion = sightlines_per_bin // numions #any remainder here goed into the remainder accounting
    final_sample_size = numbins * sightlines_per_bin
    collected_sample_size = len(chosen)
    sightlines_per_ion = final_sample_size // numions 
    maxaddtobin = sightlines_per_bin_per_ion

    doselectionloop = True
    while(doselectionloop):
        # outer loop: over ions, independent of each other
        for dataind in range(numions):
            print("Doing file %i"%dataind)
            remainder_ion = sightlines_per_ion - numbins * sightlines_per_bin_per_ion 
            print("initial remainder: %i"%remainder_ion)
            for binind in range(numbins-1, -1, -1): 
                sample_bin = set(indslist[dataind][binind]) - chosens[dataind]  # sightlines in this bin that have not been selected yet for this ion
                targetnumsightlines_sub = sightlines_per_bin_per_ion + min(remainder_ion, maxaddtobin) # number of sightlines we want from this bin   
                if targetnumsightlines_sub == 0: # on non-first loops: remainder used up, no baseline number to select
                    print('Out of sightlines for file %i'%dataind)
                    break      
                # if the bin only has a small number of sightlines, use them all
                if len(sample_bin) <= targetnumsightlines_sub:
                    chosens[dataind] |= sample_bin
                    remainder_ion += sightlines_per_bin_per_ion - len(sample_bin) # negative difference if some of the new sightlines are coming out of the remainder
                # otherwise, take a random sample
                else:
                    sample_temp = set(random.sample(sample_bin, targetnumsightlines_sub))
                    chosens[dataind] |= sample_temp
                    remainder_ion += sightlines_per_bin_per_ion - targetnumsightlines_sub # negative difference if some of the new sightlines are coming out of the remainder
            print("final remainder: %i"%remainder_ion)
        # get total sample set, check size
        for sub in chosens: 
            chosen |= sub
        collected_sample_size = len(chosen)
        remainder_total = final_sample_size - collected_sample_size
        print("Total remainder: %i"%remainder_total)
        # as long as there is enough to divide over the ions uniformly, redo the bin loop
        if remainder_total > numions - 1: # can still divide more of the sightlines
            sightlines_per_ion = remainder_total // numions
            sightlines_per_bin_per_ion = sightlines_per_ion // numbins # may be zero: in this case, just the remainder is divided up
            maxaddtobin = max(sightlines_per_bin_per_ion, 1) #if sightlines_per_bin_per_ion=0, still want to add to bins
        else: # end the while loop
            doselectionloop = False  

    # if there is a remainder left, just add some random sightlines (specwizard needs # sightlines to be a multiple of # cores, so we need to hit the exact target number)
    if remainder_total > 0:
        notchosen = set([inds for binind in range(numbins) for inds in indslist[0][binind]]) - chosen
        chosen_remainder = random.sample(notchosen, remainder_total)
    else:
        chosen_remainder = set()

    chosens = [np.array(list(chosen_sub)) for chosen_sub in chosens]
    chosen = np.array(list(chosen) + list(chosen_remainder))
    chosen_remainder = np.array(list(chosen_remainder))
    # save metadata
    selection = mdfile.create_group('Selection') 
    for i in range(numions):
        group = mdfile.create_group('Selection/file%i'%i)
        group.attrs.create('filename', files[i])
        group.create_dataset('bins',data = binss[i])
        group.create_dataset('selected_pixels_thision', data=np.array([chosens[i]/numpix,chosens[i]%numpix]).T)
        group.create_dataset('selected_values_thision', data=np.array([datas[i][ch/numpix,ch%numpix] for ch in chosens[i]]))
        group.create_dataset('selected_values_allions', data=np.array([datas[i][ch/numpix,ch%numpix] for ch in chosen]))
        group.create_dataset('sample_pixels_per_bin', data=np.array([len(indslist[i][binind]) for binind in range(numbins)]))    
    selection.create_dataset('selected_pixels_allions', data=np.array([chosen/numpix,chosen%numpix]).T)
    group = mdfile.create_group('Selection/remainder')
    group.create_dataset('selected_pixels', data=np.array([chosen_remainder/numpix,chosen_remainder%numpix]).T)  
    mdfile.close()
    
    # make los file for specwizard
    fil = open(losdir + name + '.txt','w')
    fil.write('%i\n'%(len(chosen)))
    chosen = list(chosen)
    for i in range(len(chosen)):
        fil.write('%f\t%f\t0\n'%((chosen[i]/numpix+0.5)/(float(numpix)), (chosen[i]%numpix+0.5)/(float(numpix)))) 
    fil.close()

    return None


def plotsample3_o678():
    df = h5py.File('/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample3_o6-o7-o8_L0100N1504_data.hdf5')
    fontsize = 12

    # o6
    o6bins = np.array(df['Selection/file2/bins'])
    o6sel = np.array(df['Selection/file2/selected_values_thision'])
    o6all = np.array(df['Selection/file2/selected_values_allions'])
    o6base = np.array(df['Selection/file2/sample_pixels_per_bin'])

    plt.subplot(131)

    plt.hist(o6all,bins=o6bins,log=True,label='total O6 selection',color = 'yellowgreen')
    plt.hist(o6sel,bins=o6bins,log=True,label='O6 selection by O6',color = 'green')
    plt.step(o6bins,np.array(list(o6base) + [0]), where = 'post', color='darkolivegreen', label = 'available sightlines') 

    plt.xlabel(r'$\log_{10} N_{\mathrm{O VI}} \, [\mathrm{cm}^{-2}]$',fontsize=fontsize)  
    plt.ylabel('Number of sightlines',fontsize=fontsize)
    plt.title('O VI selection',fontsize=fontsize)
    plt.legend()

    # o7
    o7bins = np.array(df['Selection/file0/bins'])
    o7sel = np.array(df['Selection/file0/selected_values_thision'])
    o7all = np.array(df['Selection/file0/selected_values_allions'])
    o7base = np.array(df['Selection/file0/sample_pixels_per_bin'])
    o7min = np.min(o7all)
    print('Min. o7 value: %s'%o7min)
    if o7min < o7bins[0]:
        o7diff = np.average(np.diff(o7bins))
        o7ext  = np.arange(o7bins[0], o7min - o7diff, -o7diff)
        o7bins = np.array(list(o7ext[1::-1]) + list(o7bins))

    plt.subplot(132)

    plt.hist(o7all,bins=o7bins,log=True,label='total O7 selection',color = 'lightcoral')
    plt.hist(o7sel,bins=o7bins,log=True,label='O7 selection by O7',color = 'red')
    plt.step(o7bins,np.array(list(o7base) + [0]), where = 'post', color='firebrick', label = 'available sightlines') 

    plt.xlabel(r'$\log_{10} N_{\mathrm{O VII}} \, [\mathrm{cm}^{-2}]$',fontsize=fontsize)  
    plt.ylabel('Number of sightlines',fontsize=fontsize)
    plt.title('O VII selection',fontsize=fontsize)
    plt.legend()

    # o8
    o8bins = np.array(df['Selection/file1/bins'])
    o8sel = np.array(df['Selection/file1/selected_values_thision'])
    o8all = np.array(df['Selection/file1/selected_values_allions'])
    o8base = np.array(df['Selection/file1/sample_pixels_per_bin'])
    o8min = np.min(o8all)
    print('Min. o8 value: %s'%o8min)
    if o8min < o8bins[0]:
        o8diff = np.average(np.diff(o8bins))
        o8ext  = np.arange(o8bins[0], o8min - o8diff, -o8diff)
        o8bins = np.array(list(o8ext[1::-1]) + list(o8bins))

    plt.subplot(133)

    plt.hist(o8all, bins=o8bins, log=True, label='total O8 selection', color='cyan')
    plt.hist(o8sel, bins=o8bins, log=True, label='O8 selection by O8', color='blue')
    plt.step(o8bins, np.array(list(o8base) + [0]), where='post', color='darkcyan', label='available sightlines') 

    plt.xlabel(r'$\log_{10} N_{\mathrm{O VIII}} \, [\mathrm{cm}^{-2}]$',fontsize=fontsize)  
    plt.ylabel('Number of sightlines',fontsize=fontsize)
    plt.title('O VIII selection',fontsize=fontsize)
    plt.legend()

    #plt.savefig('/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample3_o6-o7-o8_L0100N1504_data.png') 


def plotsample3verf_o78(kind='check19'):
    if kind == 'check19':
        df = h5py.File('/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample3-verf_o7-o8_L0100N1504_snap19_data.hdf5')
        name = '/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample3-verf_o7-o8_L0100N1504_snap19_data.png'
    elif kind == 'sample4_19':
        df = h5py.File('/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample4_o7-o8_L0100N1504_snap19_data.hdf5')
        name = '/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample4_o7-o8_L0100N1504_snap19_data.png'
    elif kind == 'sample5_26':
        df = h5py.File('/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample5_o7-o8_L0100N1504_snap26_data.hdf5')
        name = '/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample5_o7-o8_L0100N1504_snap26_data.png'
    
    fontsize = 12

    # o7
    o7bins = np.array(df['Selection/file0/bins'])
    o7sel = np.array(df['Selection/file0/selected_values_thision'])
    o7all = np.array(df['Selection/file0/selected_values_allions'])
    o7base = np.array(df['Selection/file0/sample_pixels_per_bin'])
    o7min = np.min(o7all)
    o7bins_o = np.copy(o7bins)
    print('Min. o7 value: %s'%o7min)
    if o7min < o7bins[0]:
        o7diff = np.average(np.diff(o7bins))
        o7ext  = np.arange(o7bins[0], o7min - o7diff, -o7diff)
        o7bins = np.array(list(o7ext[1::-1]) + list(o7bins))

    plt.subplot(121)

    plt.hist(o7all,bins=o7bins,log=True,label='total O7 selection',color = 'lightcoral')
    plt.hist(o7sel,bins=o7bins,log=True,label='O7 selection by O7',color = 'red')
    plt.step(o7bins_o,np.array(list(o7base) + [0]), where = 'post', color='firebrick', label = 'available sightlines') 

    plt.xlabel(r'$\log_{10} N_{\mathrm{O VII}} \, [\mathrm{cm}^{-2}]$',fontsize=fontsize)  
    plt.ylabel('Number of sightlines',fontsize=fontsize)
    plt.title('O VII selection',fontsize=fontsize)
    plt.legend()

    # o8
    o8bins = np.array(df['Selection/file1/bins'])
    o8sel = np.array(df['Selection/file1/selected_values_thision'])
    o8all = np.array(df['Selection/file1/selected_values_allions'])
    o8base = np.array(df['Selection/file1/sample_pixels_per_bin'])
    o8min = np.min(o8all)
    o8bins_o = np.copy(o8bins)
    print('Min. o8 value: %s'%o8min)
    if o8min < o8bins[0]:
        o8diff = np.average(np.diff(o8bins))
        o8ext  = np.arange(o8bins[0], o8min - o8diff, -o8diff)
        o8bins = np.array(list(o8ext[1::-1]) + list(o8bins))

    plt.subplot(122)

    plt.hist(o8all,bins=o8bins,log=True,label='total O8 selection',color = 'cyan')
    plt.hist(o8sel,bins=o8bins,log=True,label='O8 selection by O8',color = 'blue')
    plt.step(o8bins_o,np.array(list(o8base) + [0]), where = 'post', color='darkcyan', label = 'available sightlines') 

    plt.xlabel(r'$\log_{10} N_{\mathrm{O VIII}} \, [\mathrm{cm}^{-2}]$',fontsize=fontsize)  
    plt.ylabel('Number of sightlines',fontsize=fontsize)
    plt.title('O VIII selection',fontsize=fontsize)
    plt.legend()

    plt.savefig(name)


def generate_sample3_T4EOStest_subsample(specdata=None):
    # from sample for the z=0.1 O6/O7/O8-selected specwizard run for EW from column density (sample3)
    # was run with T4EOS off, effectively
    # rerun with T4EOS, looking for differences
    # run for 16 sightlines selected on each of 16 ions
    # retrieve column densities from previous specwizard run rather than from 2d maps 

    if specdata is None:
        import specwiz_proc as sp
        specdata = sp.Specout('/cosma5/data/dp004/dc-wije1/specwiz/sample3/spec.snap_027_z000p101.0.hdf5',getall=False) # we only need column densities, no need to do spectrum processing.
        specdata.getcoldens() 
    # specdata.coldens[ion] = array of #sightlines              ; log10 col. dens. values
    # specdata.positions    = array of #sightline x (x,y) values; position in cMpc/h units
    # specdata.ions         = array of #ions                    ; ion names; keys for coldens dict
    print('Specout instance loaded')
    
    name = 'los_sample3_T4EOS_subsample_test_L0100N1504_27'

    files = '/cosma5/data/dp004/dc-wije1/specwiz/sample3/spec.snap_027_z000p101.0.hdf5'
    file_open = h5py.File(files,'r')
    xfracs = np.array(file_open['Projection/x_fraction_array'])
    yfracs = np.array(file_open['Projection/y_fraction_array'])
    file_open.close()
    fracpos = np.array([xfracs,yfracs]).T

    datas = [specdata.coldens[ion] for ion in specdata.ions]
    maxcols = [np.max(specdata.coldens[ion]) for ion in specdata.ions]
    mincols = [max(12.,np.min(specdata.coldens[ion])) for ion in specdata.ions]
    numbins = 16
    sightlines_per_bin = 32
    #numpix = 32000 #used to convert index -> position
    
    maxaddtobin = sightlines_per_bin # max number of remainder sightlines to add to a single bin (first round) 
    # (number of bins needs to be fixed to guarantee divisibility by the number of cores specwizard is run on)

    numions = len(specdata.coldens)
    binss = [np.arange(mincols[i],1.001*maxcols[i],(1.0001*maxcols[i]-mincols[i])/float(numbins)) for i in range(numions)] # generate numbins+1 edges values -> numbins bins

    print('Creating hdf5 file')
    # store metadata
    mdfile = h5py.File(losdir + name + '_data.hdf5', 'w')
    header = mdfile.create_group('Header')
    header.attrs.create('input_files', np.array(files))
    header.attrs.create('los_file', losdir + name + '.txt')
    header.attrs.create('numbins', numbins)
    header.attrs.create('total_sightlines_per_bin', sightlines_per_bin)
    header.attrs.create('min_values', mincols)
    header.attrs.create('max_values', maxcols)
    header.attrs.create('samplesize', specdata.positions.shape[0])   
    header.create_dataset('bins', data = binss)
    print('Metadata stored')

    binindss = [np.digitize(datas[i],binss[i]) for i in range(len(datas))]
    # 1d dataset-> no need to flatten indices
    indslist = [[np.where(binindss[dataind] == i+1)[0] for i in range(numbins)] for dataind in range(numions)]
    #indslist = [[ indslist[dataind][binind][:,0]*numpix + indslist[dataind][binind][:,1] for binind in range(numbins)] for dataind in range(numions)] #set cannot contain arrays -> change to tuples in innermost nest; list of flattened indices corresponding in each bin for each ion
    del binindss
    chosen = set()
    chosens = [set() for data in datas]
    #bins_allchosen = [[] for dataind in range(len(datas))] # track where the sampling is complete for each data set
        
    sightlines_per_bin_per_ion = sightlines_per_bin/numions #any remainder here goed into the remainder accounting
    final_sample_size = numbins*sightlines_per_bin
    collected_sample_size = len(chosen)
    sightlines_per_ion = final_sample_size/numions 
    maxaddtobin = sightlines_per_bin_per_ion

    print('Starting selection loop')
    doselectionloop = True
    while(doselectionloop):
        # outer loop: over ions, independent of each other
        for dataind in range(numions):
            print("Doing file %i"%dataind)
            remainder_ion = sightlines_per_ion - numbins*sightlines_per_bin_per_ion 
            print("initial remainder: %i"%remainder_ion)
            for binind in range(numbins-1,-1,-1): 
                print("Doing bin %i"%binind)
                sample_bin = set(list(indslist[dataind][binind])) - chosens[dataind]  # sightlines in this bin that have not been selected yet for this ion
                targetnumsightlines_sub = sightlines_per_bin_per_ion + min(remainder_ion, maxaddtobin) # number of sightlines we want from this bin   
                if targetnumsightlines_sub == 0: # on non-first loops: remainder used up, no baseline number to select
                    print('Out of sightlines for file %i'%dataind)
                    break      
                # if the bin only has a small number of sightlines, use them all
                if len(sample_bin) <= targetnumsightlines_sub:
                    chosens[dataind] |= sample_bin
                    remainder_ion += sightlines_per_bin_per_ion - len(sample_bin) # negative difference if some of the new sightlines are coming out of the remainder
                # otherwise, take a random sample
                else:
                    sample_temp = set(random.sample(sample_bin,targetnumsightlines_sub))
                    chosens[dataind] |= sample_temp
                    remainder_ion += sightlines_per_bin_per_ion - targetnumsightlines_sub # negative difference if some of the new sightlines are coming out of the remainder
            print("final remainder: %i"%remainder_ion)
        # get total sample set, check size
        for sub in chosens: 
            chosen |= sub
        collected_sample_size = len(chosen)
        remainder_total = final_sample_size - collected_sample_size
        print("Total remainder: %i"%remainder_total)
        # as long as there is enough to divide over the ions uniformly, redo the bin loop
        if remainder_total > numions-1: # can still divide more of the sightlines
            sightlines_per_ion = remainder_total/numions
            sightlines_per_bin_per_ion = sightlines_per_ion/numbins # may be zero: in this case, just the remainder is divided up
            maxaddtobin = max(sightlines_per_bin_per_ion, 1) #if sightlines_per_bin_per_ion=0, still want to add to bins
        else: # end the while loop
            doselectionloop = False  

    # if there is a remainder left, just add some random sightlines
    if remainder_total > 0:
        notchosen = set([ind for binind in range(numbins) for ind in indslist[0][binind]]) - chosen
        chosen_remainder = random.sample(notchosen,remainder_total)
    else:
        chosen_remainder = set()

    chosens = [np.array(list(chosen_sub)) for chosen_sub in chosens]
    chosen = np.array(list(chosen))
    chosen_remainder = np.array(list(chosen_remainder))
    chosen_all = np.array(list(chosen) + list(chosen_remainder)) ########### save this !!!! ###########

    # save slection data
    print('Saving selection hdf5')
    selection = mdfile.create_group('Selection') 
    for i in range(numions):
        group = mdfile.create_group('Selection/%s'%(specdata.ions[i]))
        group.attrs.create('ion', np.array(specdata.ions[i].encode('utf8') )) # default unicode is not well-supported by h5py
        group.create_dataset('bins',data = binss[i])
        group.create_dataset('selected_fracpos_thision', data=fracpos[chosens[i]])
        group.create_dataset('selected_values_thision', data=datas[i][chosens[i]])
        group.create_dataset('selected_values_allions', data=datas[i][chosen_all])
        group.create_dataset('sample_pixels_per_bin', data=np.array([len(indslist[i][binind]) for binind in range(numbins)]))    
    selection.create_dataset('selected_fracpos_allions', data=fracpos[chosen_all])
    group = mdfile.create_group('Selection/remainder')
    group.create_dataset('selected_fracpos', data=fracpos[chosen_remainder])  
    mdfile.close()
    print('done. Saving los file')
    # make los file for specwizard
    fil = open(losdir + name + '.txt','w')
    fil.write('%i\n'%(len(chosen_all)))
    #chosen = list(chosen)
    for i in range(len(chosen_all)):
        fil.write('%f\t%f\t0\n'%(fracpos[chosen_all[i],0],fracpos[chosen_all[i],1])) 
    fil.close()
    print('done')

    return None


def subplotsample3_T4EOSsub(ax, df, ion, ionlab, fontsize=12, doylabel = True):
    print(ion)
    bins = np.array(df['Selection/%s/bins'%ion])
    sel = np.array(df['Selection/%s/selected_values_thision'%ion])
    ionall = np.array(df['Selection/%s/selected_values_allions'%ion])
    base = np.array(df['Selection/%s/sample_pixels_per_bin'%ion])

    ax.hist(ionall,bins=bins,log=True,label='total %s selection'%ion,color = 'yellowgreen')
    ax.hist(sel,bins=bins,log=True,label='%s selection by %s'%(ion,ion),color = 'green')
    ax.step(bins,np.array(list(base) + [0]), where = 'post', color='darkolivegreen', label = 'available sightlines') 

    ax.set_xlabel(r'$\log_{10} N_{\mathrm{%s}} \, [\mathrm{cm}^{-2}]$'%ionlab,fontsize=fontsize)  
    ax.set_ylabel('Number of sightlines',fontsize=fontsize)
    #ax.set_title('%s selection'%ionlab,fontsize=fontsize)
    ax.legend(fontsize=fontsize)

def plotsample3_T4EOSsub():
    df = h5py.File('/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample3_T4EOS_subsample_test_L0100N1504_27_data.hdf5')
    fontsize = 12

    fig, axes = plt.subplots(figsize = (20.,20.), ncols=4,nrows=4)

    ions = ['c3', 'c4', 'c5', 'c6', 'n6', 'n7', 'ne8', 'ne9', 'fe17', 'si3', 'si4', 'o4', 'o5', 'o6', 'o7', 'o8']
    ionlabs = ['C III', 'C IV', 'C V', 'C VI', 'N VI', 'N VII', 'Ne VIII', 'Ne IX', 'Fe XVII', 'Si III', 'Si IV', 'O IV', 'O V', 'O VI', 'O VII', 'O VIII']
    
    for i in range(16):
        xi = i/4
        yi = i%4
        subplotsample3_T4EOSsub(axes[xi,yi], df, ions[i], ionlabs[i], fontsize=fontsize, doylabel = (yi==0))
         
    plt.savefig('/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample3_T4EOS_subsample_test_L0100N1504_27_data.png',format = 'png',bbox_inches = 'tight') 


def getmatch_2dprojections_sample3():
    '''
    for the filenames and fills below, get the 2d projection 
    quantities at each pixel in the order of the output file
    checked: the pixels in the sample selection file match the
    x/y fractions in the specwizard output file (incl. order)
    -> can use the selection file to get the pixel selection

    selection was from L0100N1504, snapshot 27
    '''
    fills = [str(float(i)) for i in (np.arange(16)/16.+1/32.)*100.]
    filenames_2d = []
    groupnames_2d = []
    filename_samplesel = '/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample3_o6-o7-o8_L0100N1504_data.hdf5'
    filename_out = 'dummy.hdf5'    

    selfile = h5py.File(filename_samplesel, 'r')
    basesel = np.array(selfile['Selection/selected_pixels_allions'])
    sel = (basesel[:,0], basesel[:,1]) # format for fancy indexing of arrays
    selfile.close()

    outfile = h5py.File(filename_out, 'w')  

    numspectra = len(sel[0])
    numslices  = len(fills)
    
    # store header data
    header = outfile.create_group['Header']
    header.attrs.create('sample_file', np.array(selfile.encode('utf8')) ) # default unicode is not well-supported by h5py
    ds = header.create_dataset('xy_fractions', basesel)
    ds.attrs.create('units', np.array('fraction of box size'.encode('utf8')))
    ds.attrs.create('news', np.array('retrieved from sample_file'.encode('utf8')))
    ds = header.create_dataset('zcen_values', np.array(fills))
    ds.attrs.create('news', 'values to put in the place of <value> to retrieve file names')
    ds.close()
    header.close()

    for ind in range(len(filenames)*len(fills)):
        fileind = ind/len(fills)
        fillind = ind%len(fills)
        print('Doing file %i, fill %i'%(fileind, fillind))

        if fillind == 0: # set file name, create group
            filename = filenames[fileind]
            groupname = groupnames[fileind]

            filename_store = ((filename.split['/'])[0])%('<value>') # no need to store directories
            group = outfile.create_group(groupname)
            group.attrs.create('filename', np.array(filename_store.encode('utf8') )) # default unicode is not well-supported by h5py
            ds = group.create_dataset('values', (numspectra, numslices))
       
        data = np.load(filename%fills[fillind])['arr_0']
        ds[:,fillind] = data[sel]
        del data

        if fillind == numslices-1: # close group
            group.close()
    
    outfile.close()
    return outfilename


def generate_unisample(samplename='sample3'):
    if samplename == 'sample3':
        # sample for the z=0.1 O6/O7/O8-selected specwizard run for EW from column density (sample3)
        file1 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz'
        file2 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz'
        file3 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        data1 = np.load(file1)['arr_0']
        data2 = np.load(file2)['arr_0']
        data3 = np.load(file3)['arr_0']
        name = 'los_sample3_o6-o7-o8_L0100N1504'
    
        datas = [data1, data2, data3]
        files = [file1, file2, file3]
        maxcols = [np.max(data) for data in datas]
        mincols = [13., 13., 13.]
        numbins = 128
        sightlines_per_bin = 128
        numpix = 32000 #used to convert index -> position
        
        maxaddtobin = sightlines_per_bin # max number of remainder sightlines to add to a single bin (first round) 
        # (number of bins needs to be fixed to guarantee divisibility by the number of cores specwizard is run on)
    elif samplename == 'sample6':
        losdir = '/net/luttero/data2/specwizard_data/sample6/' # changed output directory for Leiden
        file1 = '/net/quasar/data2/wijers/temp/coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz'
        file2 = '/net/quasar/data2/wijers/temp/coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        file3 = '/net/quasar/data2/wijers/temp/coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        data1 = np.load(file1)['arr_0']
        data2 = np.load(file2)['arr_0']
        data3 = np.load(file3)['arr_0']
        name = 'los_sample6_ne8-ne9-fe17_L0100N1504'
    
        datas = [data1, data2, data3]
        files = [file1, file2, file3]
        maxcols = [np.max(data) for data in datas]
        mincols = [13., 13., 13.]
        numbins = 128
        sightlines_per_bin = 128
        numpix = 32000 #used to convert index -> position
        
        maxaddtobin = sightlines_per_bin # max number of remainder sightlines to add to a single bin (first round) 
        # (number of bins needs to be fixed to guarantee divisibility by the number of cores specwizard is run on)
    
    elif samplename == 'sample7':  
        # sample for the z=0.5 O7/O8-selected specwizard run for EW from column density
        file1 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o7_L0100N1504_23_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        file2 = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/coldens_o8_L0100N1504_23_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
        data1 = np.load(file1)['arr_0']
        data2 = np.load(file2)['arr_0']
        name = 'los_sample7_o7-o8_L0100N1504'
    
        datas = [data1, data2]
        files = [file1, file2]
        maxcols = [np.max(data) for data in datas]
        mincols = [13., 13.]
        numbins = 128
        sightlines_per_bin = 128
        numpix = 32000 #used to convert index -> position


    # below this line: everything should allow modification of the above, including number of data sets
    numions = len(datas)
    if numbins*sightlines_per_bin > numpix**2:
        print('You cannot select %i sightlines from a sample of %i pixels.'%(numbins*sightlines_per_bin, numpix**2))
        return None
    binss = [np.arange(mincols[i],1.001*maxcols[i],(1.0001*maxcols[i]-mincols[i])/float(numbins)) for i in range(numions)] # generate numbins+1 edges values -> numbins bins
    # store metadata
    mdfile = h5py.File(losdir + name + '_data.hdf5', 'w')
    header = mdfile.create_group('Header')
    header.attrs.create('input_files', np.array(files))
    header.attrs.create('los_file', losdir + name + '.txt')
    header.attrs.create('numbins', numbins)
    header.attrs.create('total_sightlines_per_bin', sightlines_per_bin)
    header.attrs.create('min_values', mincols)
    header.attrs.create('max_values', maxcols)
    header.attrs.create('numpix', numpix)   
    header.create_dataset('bins', data = binss)

    binindss = [np.digitize(datas[i],binss[i]) for i in range(len(datas))]
    indslist = [[np.array(np.where(binindss[dataind] == i+1)).T for i in range(numbins)] for dataind in range(numions)]
    indslist = [[ indslist[dataind][binind][:,0]*numpix + indslist[dataind][binind][:,1] for binind in range(numbins)] for dataind in range(numions)] #set cannot contain arrays -> change to tuples in innermost nest; list of flattened indices corresponding in each bin for each ion
    del binindss
    chosen = set()
    chosens = [set() for data in datas]
    #bins_allchosen = [[] for dataind in range(len(datas))] # track where the sampling is complete for each data set
        
    sightlines_per_bin_per_ion = sightlines_per_bin/numions #any remainder here goed into the remainder accounting
    final_sample_size = numbins*sightlines_per_bin
    collected_sample_size = len(chosen)
    sightlines_per_ion = final_sample_size/numions 
    maxaddtobin = sightlines_per_bin_per_ion

    doselectionloop = True
    while(doselectionloop):
        # outer loop: over ions, independent of each other
        for dataind in range(numions):
            print("Doing file %i"%dataind)
            remainder_ion = sightlines_per_ion - numbins*sightlines_per_bin_per_ion 
            print("initial remainder: %i"%remainder_ion)
            for binind in range(numbins-1,-1,-1): 
                sample_bin = set(indslist[dataind][binind]) - chosens[dataind]  # sightlines in this bin that have not been selected yet for this ion
                targetnumsightlines_sub = sightlines_per_bin_per_ion + min(remainder_ion, maxaddtobin) # number of sightlines we want from this bin   
                if targetnumsightlines_sub == 0: # on non-first loops: remainder used up, no baseline number to select
                    print('Out of sightlines for file %i'%dataind)
                    break      
                # if the bin only has a small number of sightlines, use them all
                if len(sample_bin) <= targetnumsightlines_sub:
                    chosens[dataind] |= sample_bin
                    remainder_ion += sightlines_per_bin_per_ion - len(sample_bin) # negative difference if some of the new sightlines are coming out of the remainder
                # otherwise, take a random sample
                else:
                    sample_temp = set(random.sample(sample_bin,targetnumsightlines_sub))
                    chosens[dataind] |= sample_temp
                    remainder_ion += sightlines_per_bin_per_ion - targetnumsightlines_sub # negative difference if some of the new sightlines are coming out of the remainder
            print("final remainder: %i"%remainder_ion)
        # get total sample set, check size
        for sub in chosens: 
            chosen |= sub
        collected_sample_size = len(chosen)
        remainder_total = final_sample_size - collected_sample_size
        print("Total remainder: %i"%remainder_total)
        # as long as there is enough to divide over the ions uniformly, redo the bin loop
        if remainder_total > numions-1: # can still divide more of the sightlines
            sightlines_per_ion = remainder_total/numions
            sightlines_per_bin_per_ion = sightlines_per_ion/numbins # may be zero: in this case, just the remainder is divided up
            maxaddtobin = max(sightlines_per_bin_per_ion, 1) #if sightlines_per_bin_per_ion=0, still want to add to bins
        else: # end the while loop
            doselectionloop = False  

    # if there is a remainder left, just add some random sightlines
    if remainder_total > 0:
        notchosen = set([indslist[0][binind] for binind in range(numbins)]) - chosen
        chosen_remainder = random.sample(notchosen,remainder_total)
    else:
        chosen_remainder = set()

    chosens = [np.array(list(chosen_sub)) for chosen_sub in chosens]
    chosen = np.array(list(chosen))
    chosen_remainder = np.array(list(chosen_remainder))
    # save metadata
    selection = mdfile.create_group('Selection') 
    for i in range(numions):
        group = mdfile.create_group('Selection/file%i'%i)
        group.attrs.create('filename', files[i])
        group.create_dataset('bins',data = binss[i])
        group.create_dataset('selected_pixels_thision', data=np.array([chosens[i]/numpix,chosens[i]%numpix]).T)
        group.create_dataset('selected_values_thision', data=np.array([datas[i][ch/numpix,ch%numpix] for ch in chosens[i]]))
        group.create_dataset('selected_values_allions', data=np.array([datas[i][ch/numpix,ch%numpix] for ch in chosen]))
        group.create_dataset('sample_pixels_per_bin', data=np.array([len(indslist[i][binind]) for binind in range(numbins)]))    
    selection.create_dataset('selected_pixels_allions', data=np.array([chosen/numpix,chosen%numpix]).T)
    group = mdfile.create_group('Selection/remainder')
    group.create_dataset('selected_pixels', data=np.array([chosen_remainder/numpix,chosen_remainder%numpix]).T)  
    mdfile.close()
    
    # make los file for specwizard
    fil = open(losdir + name + '.txt','w')
    fil.write('%i\n'%(len(chosen)))
    chosen = list(chosen)
    for i in range(len(chosen)):
        fil.write('%f\t%f\t0\n'%((chosen[i]/numpix+0.5)/(float(numpix)), (chosen[i]%numpix+0.5)/(float(numpix)))) 
    fil.close()

    return None
    
