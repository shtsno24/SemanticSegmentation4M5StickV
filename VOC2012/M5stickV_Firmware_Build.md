# 詰まったところ  

Components configuration -> Enbale micropython component -> Others -> Compile lib_sipeed_kpu source code を有効化してしまい、menuconfigが立ち上がらなくなる

--> (path_to_maixpy)/MaixPy/projects/maixpy_m5stickv/build/config/global_config.mkでも設定できるので、ここでCONFIG_COMPONENT_LIB_SIPEED_KPU_SRC_ENABLEの部分をコメントアウト。
