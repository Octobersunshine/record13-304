from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from normality_test import NormalityTest


app = FastAPI(
    title="正态性检验服务",
    description="基于 Python + SciPy 的正态性检验服务，支持 Shapiro-Wilk 检验（小样本）和 Kolmogorov-Smirnov 检验（大样本）",
    version="1.0.0"
)


class NormalityTestRequest(BaseModel):
    data: List[float] = Field(..., description="待检验的数值数组", min_length=3)
    alpha: Optional[float] = Field(0.05, description="显著性水平，默认 0.05", gt=0, lt=1)
    test_method: Optional[str] = Field(
        "auto", 
        description="检验方法：auto（自动选择）、shapiro（Shapiro-Wilk）、ks（Kolmogorov-Smirnov）、comprehensive（综合检验）"
    )


class QQPlotRequest(BaseModel):
    data: List[float] = Field(..., description="待分析的数值数组", min_length=3)


@app.get("/")
async def root():
    return {
        "service": "正态性检验服务",
        "version": "1.0.0",
        "endpoints": {
            "POST /test": "执行正态性检验",
            "POST /qq-plot": "获取 QQ 图数据",
            "GET /info": "获取服务说明"
        }
    }


@app.get("/info")
async def info():
    return {
        "service_name": "正态性检验服务",
        "supported_tests": {
            "shapiro_wilk": {
                "name": "Shapiro-Wilk 检验",
                "description": "适用于小样本数据（推荐 n < 50，最大支持 n ≤ 5000）",
                "principle": "基于数据的顺序统计量与正态分布顺序统计量的相关性",
                "strengths": "小样本下检验效能较高"
            },
            "kolmogorov_smirnov": {
                "name": "Kolmogorov-Smirnov 检验",
                "description": "适用于大样本数据（推荐 n ≥ 50）",
                "principle": "基于经验分布函数与理论正态分布函数的最大差异",
                "strengths": "对分布形状敏感，不受参数估计影响"
            }
        },
        "auto_selection_rule": "当样本量 ≤ 5000 时自动选择 Shapiro-Wilk，> 5000 时自动选择 Kolmogorov-Smirnov",
        "qq_plot": {
            "name": "QQ 图数据",
            "description": "输出用于绘制 QQ 图的数据点，包括理论分位数、样本分位数和参考线",
            "endpoint": "POST /qq-plot"
        },
        "default_alpha": 0.05
    }


@app.post("/test")
async def perform_test(request: NormalityTestRequest):
    try:
        tester = NormalityTest(alpha=request.alpha)
        
        method = request.test_method.lower()
        
        if method == "auto":
            result = tester.auto_test(request.data)
        elif method in ["shapiro", "shapiro-wilk", "shapiro_wilk"]:
            result = tester.shapiro_wilk_test(request.data)
        elif method in ["ks", "kolmogorov-smirnov", "kolmogorov_smirnov"]:
            result = tester.kolmogorov_smirnov_test(request.data)
        elif method == "comprehensive":
            result = tester.comprehensive_test(request.data)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的检验方法: {request.test_method}。请使用: auto, shapiro, ks, comprehensive"
            )
        
        return {
            "success": True,
            "result": result
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检验失败: {str(e)}")


@app.post("/qq-plot")
async def qq_plot(request: QQPlotRequest):
    try:
        tester = NormalityTest()
        result = tester.qq_plot_data(request.data)
        return {
            "success": True,
            "result": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QQ 图数据生成失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
