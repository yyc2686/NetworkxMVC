from error.CommonError import CommonError
from error.EmBusinessError import EmBusinessError


# 包装器/适配器业务异常实现类
class BusinessException(CommonError, Exception):
    # 直接接收EmBusiness的传参用于初始化业务异常
    commonError = None

    def __init__(self, commonError, errMsg=""):
        super()
        # 直接接收EmBusiness的传参用于初始化业务异常
        self.setCommonError(commonError)
        if errMsg != "":
            self.commonError.setErrMsg(errMsg)

    def getErrCode(self):
        return self.commonError.getErrCode()

    def getErrMsg(self):
        return self.commonError.getErrMsg()

    def setErrMsg(self, errMsg):
        self.commonError.setErrMsg(errMsg)
        return self

    def setCommonError(self, commonError):
        self.commonError = commonError

    def __str__(self):
        return str(self.commonError)


if __name__ == "__main__":
    a = BusinessException(EmBusinessError.PARAMETER_VALIDATION_ERROR)
    print(a)
    # print(a.__class__)
    # print(a.getErrCode())
    # print(a.getErrMsg())
    #
    # print()
    #
    # a = BusinessException(EmBusinessError.PARAMETER_VALIDATION_ERROR, "哈哈哈")
    # print(a)
    # print(a.__class__)
    # print(a.getErrCode())
    # print(a.getErrMsg())
