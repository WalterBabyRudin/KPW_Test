from logging import Handler
import numpy as np
import matplotlib.pyplot as plt


# ### Plotting for first user
# fig, axes = plt.subplots(ncols=5, figsize=(25,5))

# x = np.arange(400,855)


# PW_Guest_Test_Stat_hist = np.load('Results/PW_Guest_Test_Stat_hist.npy')
# ME_Guest_Test_Stat_hist = np.load('Results/ME_Guest_Test_Stat_hist.npy')
# MMDO_Test_Stat_hist = np.load('Results/MMDO_Test_Stat_hist.npy')
# NTK_Test_Stat_hist = np.load('Results/NTK_Test_Stat_hist.npy')
# KPW_Guest_Test_Stat_hist = np.load('Results/KPW_Guest_Test_Stat_hist.npy')


# axes[0].plot(x, NTK_Test_Stat_hist, label=r'MMD-NTK', 
#             linestyle = '-', marker = 'h', color='#BD7D74',ms=2,linewidth=1)
# axes[0].tick_params(labelsize=18)
# axes[0].axvline(500,color='black',ls=':',lw=2)
# axes[1].plot(x, ME_Guest_Test_Stat_hist, label=r'ME', 
#             linestyle = ':', marker = 'o', color='b',ms=2,linewidth=1,alpha=0.4)
# axes[1].tick_params(labelsize=18)
# axes[1].axvline(500,color='black',ls=':',lw=2)
# axes[2].plot(x, PW_Guest_Test_Stat_hist, label=r'$\mathcal{P}$W', 
#             linestyle = '-.', marker = '*', color='purple',ms=2,linewidth=1)
# axes[2].tick_params(labelsize=18)
# axes[2].axvline(500,color='black',ls=':',lw=2)
# axes[3].plot(x, MMDO_Test_Stat_hist, label=r'MMD-O', 
#             linestyle = '-.', marker = '*', color='#576690',ms=2,linewidth=1)
# axes[3].tick_params(labelsize=18)
# axes[3].axvline(500,color='black',ls=':',lw=2)
# axes[4].plot(x, KPW_Guest_Test_Stat_hist, label=r'$\mathcal{KP}$W', 
#             linestyle = '--', marker = '*', color='r',ms=2,linewidth=1)
# axes[4].tick_params(labelsize=18)
# axes[4].axvline(500,color='black',ls=':',lw=2)
# plt.subplots_adjust(top=0.84, bottom=0.15,left=0.04, wspace=0.3, right=0.99)
# fig.legend(loc="upper center",fontsize=20,ncol=5,bbox_to_anchor=(0.5,1))
# plt.savefig('Exp_detection_summary_1.pdf')





# ### Plotting for second user
# fig, axes = plt.subplots(ncols=5, figsize=(25,5))

# x = np.arange(400,855)


# PW_Guest_Test_Stat_hist = np.load('Results/PW_Guest_Test_Stat_hist_2.npy')
# ME_Guest_Test_Stat_hist = np.load('Results/ME_Guest_Test_Stat_hist_2.npy')
# MMDO_Test_Stat_hist = np.load('Results/MMDO_Test_Stat_hist_2.npy')
# NTK_Test_Stat_hist = np.load('Results/NTK_Test_Stat_hist_2.npy')
# KPW_Guest_Test_Stat_hist = np.load('Results/KPW_Guest_Test_Stat_hist_2.npy')


# axes[0].plot(x, NTK_Test_Stat_hist, #label=r'MMD-NTK', 
#             linestyle = '-', marker = 'h', color='#BD7D74',ms=2,linewidth=1)
# axes[0].tick_params(labelsize=18)
# axes[0].axvline(500,color='black',ls=':',lw=2)
# axes[1].plot(x, ME_Guest_Test_Stat_hist, #label=r'ME', 
#             linestyle = ':', marker = 'o', color='b',ms=2,linewidth=1,alpha=0.4)
# axes[1].tick_params(labelsize=18)
# axes[1].axvline(500,color='black',ls=':',lw=2)
# axes[2].plot(x, PW_Guest_Test_Stat_hist, #label=r'$\mathcal{P}$W', 
#             linestyle = '-.', marker = '*', color='purple',ms=2,linewidth=1)
# axes[2].tick_params(labelsize=18)
# axes[2].axvline(500,color='black',ls=':',lw=2)
# axes[3].plot(x, MMDO_Test_Stat_hist, #label=r'MMD-O', 
#             linestyle = '-.', marker = '*', color='#576690',ms=2,linewidth=1)
# axes[3].tick_params(labelsize=18)
# axes[3].axvline(500,color='black',ls=':',lw=2)
# axes[4].plot(x, KPW_Guest_Test_Stat_hist, #label=r'$\mathcal{KP}$W', 
#             linestyle = '--', marker = '*', color='r',ms=2,linewidth=1)
# axes[4].tick_params(labelsize=18)
# axes[4].axvline(500,color='black',ls=':',lw=2)
# plt.subplots_adjust(top=0.84, bottom=0.15,left=0.04, wspace=0.3, right=0.99)
# plt.savefig('Exp_detection_summary_2.pdf')


# ### Plotting for third user
# fig, axes = plt.subplots(ncols=5, figsize=(25,5))

# x = np.arange(400,746)


# PW_Guest_Test_Stat_hist = np.load('Results/PW_Guest_Test_Stat_hist_3.npy')
# ME_Guest_Test_Stat_hist = np.load('Results/ME_Guest_Test_Stat_hist_3.npy')
# MMDO_Test_Stat_hist = np.load('Results/MMDO_Test_Stat_hist_3.npy')
# NTK_Test_Stat_hist = np.load('Results/NTK_Test_Stat_hist_3.npy')
# KPW_Guest_Test_Stat_hist = np.load('Results/KPW_Guest_Test_Stat_hist_3.npy')


# axes[0].plot(x, NTK_Test_Stat_hist, #label=r'MMD-NTK', 
#             linestyle = '-', marker = 'h', color='#BD7D74',ms=2,linewidth=1)
# axes[0].tick_params(labelsize=18)
# axes[0].axvline(500,color='black',ls=':',lw=2)
# axes[1].plot(x, ME_Guest_Test_Stat_hist, #label=r'ME', 
#             linestyle = ':', marker = 'o', color='b',ms=2,linewidth=1,alpha=0.4)
# axes[1].tick_params(labelsize=18)
# axes[1].axvline(500,color='black',ls=':',lw=2)
# axes[2].plot(x, PW_Guest_Test_Stat_hist, #label=r'$\mathcal{P}$W', 
#             linestyle = '-.', marker = '*', color='purple',ms=2,linewidth=1)
# axes[2].tick_params(labelsize=18)
# axes[2].axvline(500,color='black',ls=':',lw=2)
# axes[3].plot(x, MMDO_Test_Stat_hist, #label=r'MMD-O', 
#             linestyle = '-.', marker = '*', color='#576690',ms=2,linewidth=1)
# axes[3].tick_params(labelsize=18)
# axes[3].axvline(500,color='black',ls=':',lw=2)
# axes[4].plot(x, KPW_Guest_Test_Stat_hist, #label=r'$\mathcal{KP}$W', 
#             linestyle = '--', marker = '*', color='r',ms=2,linewidth=1)
# axes[4].tick_params(labelsize=18)
# axes[4].axvline(500,color='black',ls=':',lw=2)
# plt.subplots_adjust(top=0.84, bottom=0.15,left=0.04, wspace=0.3, right=0.99)
# plt.savefig('Exp_detection_summary_3.pdf')



# ### Plotting for third user
# fig, axes = plt.subplots(ncols=5, figsize=(25,5))

# x = np.arange(400,746)


# PW_Guest_Test_Stat_hist = np.load('Results/PW_Guest_Test_Stat_hist_3.npy')
# ME_Guest_Test_Stat_hist = np.load('Results/ME_Guest_Test_Stat_hist_3.npy')
# MMDO_Test_Stat_hist = np.load('Results/MMDO_Test_Stat_hist_3.npy')
# NTK_Test_Stat_hist = np.load('Results/NTK_Test_Stat_hist_3.npy')
# KPW_Guest_Test_Stat_hist = np.load('Results/KPW_Guest_Test_Stat_hist_3.npy')


# axes[0].plot(x, NTK_Test_Stat_hist, #label=r'MMD-NTK', 
#             linestyle = '-', marker = 'h', color='#BD7D74',ms=2,linewidth=1)
# axes[0].tick_params(labelsize=18)
# axes[0].axvline(500,color='black',ls=':',lw=2)
# axes[1].plot(x, ME_Guest_Test_Stat_hist, #label=r'ME', 
#             linestyle = ':', marker = 'o', color='b',ms=2,linewidth=1,alpha=0.4)
# axes[1].tick_params(labelsize=18)
# axes[1].axvline(500,color='black',ls=':',lw=2)
# axes[2].plot(x, PW_Guest_Test_Stat_hist, #label=r'$\mathcal{P}$W', 
#             linestyle = '-.', marker = '*', color='purple',ms=2,linewidth=1)
# axes[2].tick_params(labelsize=18)
# axes[2].axvline(500,color='black',ls=':',lw=2)
# axes[3].plot(x, MMDO_Test_Stat_hist, #label=r'MMD-O', 
#             linestyle = '-.', marker = '*', color='#576690',ms=2,linewidth=1)
# axes[3].tick_params(labelsize=18)
# axes[3].axvline(500,color='black',ls=':',lw=2)
# axes[4].plot(x, KPW_Guest_Test_Stat_hist, #label=r'$\mathcal{KP}$W', 
#             linestyle = '--', marker = '*', color='r',ms=2,linewidth=1)
# axes[4].tick_params(labelsize=18)
# axes[4].axvline(500,color='black',ls=':',lw=2)
# plt.subplots_adjust(top=0.84, bottom=0.15,left=0.04, wspace=0.3, right=0.99)
# plt.savefig('Exp_detection_summary_3.pdf')



### Plotting for last user
fig, axes = plt.subplots(ncols=5, figsize=(25,5))

x = np.arange(400,855)

PW_Guest_Test_Stat_hist = np.load('Results/PW_Guest_Test_Stat_hist_4.npy')
ME_Guest_Test_Stat_hist = np.load('Results/ME_Guest_Test_Stat_hist_4.npy')
MMDO_Test_Stat_hist = np.load('Results/MMDO_Test_Stat_hist_4.npy')
NTK_Test_Stat_hist = np.load('Results/NTK_Test_Stat_hist_4.npy')
KPW_Guest_Test_Stat_hist = np.load('Results/KPW_Guest_Test_Stat_hist_4.npy')


axes[0].plot(x, NTK_Test_Stat_hist, #label=r'MMD-NTK', 
            linestyle = '-', marker = 'h', color='#BD7D74',ms=2,linewidth=1)
axes[0].tick_params(labelsize=18)
axes[0].axvline(500,color='black',ls=':',lw=2)
axes[1].plot(x, ME_Guest_Test_Stat_hist, #label=r'ME', 
            linestyle = ':', marker = 'o', color='b',ms=2,linewidth=1,alpha=0.4)
axes[1].tick_params(labelsize=18)
axes[1].axvline(500,color='black',ls=':',lw=2)
axes[2].plot(x, PW_Guest_Test_Stat_hist, #label=r'$\mathcal{P}$W', 
            linestyle = '-.', marker = '*', color='purple',ms=2,linewidth=1)
axes[2].tick_params(labelsize=18)
axes[2].axvline(500,color='black',ls=':',lw=2)
axes[3].plot(x, MMDO_Test_Stat_hist, #label=r'MMD-O', 
            linestyle = '-.', marker = '*', color='#576690',ms=2,linewidth=1)
axes[3].tick_params(labelsize=18)
axes[3].axvline(500,color='black',ls=':',lw=2)
axes[4].plot(x, KPW_Guest_Test_Stat_hist, #label=r'$\mathcal{KP}$W', 
            linestyle = '--', marker = '*', color='r',ms=2,linewidth=1)
axes[4].tick_params(labelsize=18)
axes[4].axvline(500,color='black',ls=':',lw=2)
plt.subplots_adjust(top=0.84, bottom=0.15,left=0.04, wspace=0.3, right=0.99)
plt.savefig('Exp_detection_summary_4.pdf')