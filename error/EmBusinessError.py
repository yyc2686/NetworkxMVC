from error.CommonError import CommonError
from enum import Enum


class EmBusinessError(CommonError, Enum):
    # 通用错误类型10001开始
    PARAMETER_VALIDATION_ERROR = {"errCode": 10001, "errMsg": "参数不合法"}
    UNKNOW_ERROR = {"errCode": 10002, "errMsg": "未知错误"}

    # Patent相关错误20001开始
    # Country相关错误30001开始
    # Metric相关错误40001开始
    # Matplotlib相关错误50001开始

    # ComplexNetwork相关错误60001开始
    NETWORK_GENERATE_FAIL = {"errCode": 60001, "errMsg": "网络生成失败"}
    NETWORK_MISMATCH_ERROR = {"errCode": 60002, "errMsg": "网络类型不匹配"}
    NETWORK_LIBRARY_METRIC_ERROR = {"errCode": 60003, "errMsg": "网络库指标计算出错"}
    NETWORK_CUSTOM_METRIC_ERROR = {"errCode": 60004, "errMsg": "网络自定义指标计算出错"}

    def getErrCode(self):
        return self.value.get("errCode")

    def getErrMsg(self):
        return self.value.get("errMsg")

    def setErrMsg(self, errMsg):
        self.value["errMsg"] = errMsg
        return self

    def __str__(self):
        return str(self.value)


if __name__ == "__main__":
    a = EmBusinessError.PARAMETER_VALIDATION_ERROR
    print(a)
    print(a.getErrCode())
    print(a.getErrMsg())

    print()

    a = EmBusinessError.PARAMETER_VALIDATION_ERROR.setErrMsg("哈哈哈")
    print(a)
    print(a.getErrCode())
    print(a.getErrMsg())
