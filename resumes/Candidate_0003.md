姓名: Candidate_0003

# Candidate_0003
## 当前职位
大模型全栈开发工程师

## 职级
level 8

## 职业背景概述
拥有8年以上软件开发经验的大模型全栈开发工程师，专注于LLM Agent架构设计与开发。践行Harness Engineer理念，以系统工程化的方式构建Agent——通过精细化的上下文工程（Context Engineering）、工具链编排（Tool Orchestration）与状态流程控制驱动大模型产出高质量结果。深度掌握LangGraph工作流编排、LangChain Tool Calling、CrewAI多Agent协作等核心Agent技术栈，具备DeepAgent架构设计与RAG系统端到端构建的实战经验。精通Python/Go全栈技术，主要服务全球财富500强客户，交付了多个生产级AI Agent与RAG系统。

## 教育背景
- 武汉大学珞珈学院，2015年毕业
- 本科，电气工程及其自动化

## 早期职业背景
- 上海艾融软件股份有限公司 软件工程师（2018.07-2019.07），负责Python后端开发，担任CEPH监控项目开发组长
- 埃森哲（中国）有限公司 软件开发工程师（2019.09-2022.02），参与Botzero智能机器人平台架构设计与供应链决策平台后端开发
- 武汉中海庭数据技术有限公司 软件开发（2022.03-2023.08），担任GRB转EFD平台设计负责人，主导后端架构搭建与性能优化
- 埃森哲（中国）有限公司 AI工程师（2023.09-至今），专注大模型Agent开发与AI工程化，服务全球500强客户

## 核心技能
- AI Agent架构与开发（Harness Engineer）：践行"不只是写Prompt，而是工程化驾驭大模型"的Harness Engineer理念，精通LangGraph State Graph工作流编排（节点设计、条件分支、Conditional Edge、Human-in-the-Loop interrupt机制）、LangChain Tool Calling与自定义工具链开发、CrewAI多Agent协作（Role-Based Agent、Sequential/Hierarchical Process）；具备DeepAgent架构设计、Agent Pipeline编排及多轮对话状态管理能力
- 上下文工程（Context Engineering）：大模型输出质量的核心驱动力——精通Query Rewrite提升检索召回率、多源Retrieval结果的去重/排序/窗口裁剪编排、动态Prompt模板构建、对话历史压缩与摘要管理，确保大模型在每一步推理中获得最优上下文输入
- RAG系统构建：端到端设计与开发RAG流程，涵盖文档解析（PDF OCR/docx/Excel）、Chunk策略、Embedding向量化、多路检索（稠密向量语义检索、稀疏向量BM25全文检索、混合检索）、Reranker重排序
- 大模型工程化：Prompt Engineering（结构化提取指令设计、Few-Shot、Chain-of-Thought）、Streaming流式输出、多模型集成与调度（Azure OpenAI、通义千问、微软PHI等）
- 全栈后端开发：Python（FastAPI/Flask/Django）、Go（Gin），精通异步编程（asyncio/aiohttp/aiofiles）、微服务架构与RESTful API设计

## 行业经验
- 医药与生命科学
- 工业制造与能源
- 科技与软件服务

## 项目经验
### AI Agent与RAG平台（2025.03-至今）
**角色：** 大模型全栈开发工程师 / 技术负责人
**团队：** 与前端、产品、数据团队跨职能协作，负责Agent及后端架构的技术方案设计与核心模块开发

#### MoBao智能问答平台
- 负责平台架构设计与数据建模，基于FastAPI搭建开发框架（Alembic迁移、权限中间件、异常处理、Loguru日志、Poetry包管理）；平台上线后日均服务200+次对话，稳定支撑业务团队日常问答需求
- 设计端到端RAG Pipeline：Q&A Excel与PDF文件解析 → Chunk分片 → Embedding向量化 → Milvus入库；初期仅采用语义检索（稠密向量），发现中文专业术语召回率不足，引入Jieba中文分词+自定义医药词典实现BM25全文检索（稀疏向量），最终设计混合检索策略融合两者优势，显著提升专业领域的检索召回率；集成BGE-Reranker-Large对候选片段二次排序，进一步提升检索精度
- 采用Harness Engineer思想构建Agent系统：基于LangGraph设计多节点工作流，Router节点对用户意图分类路由（智能推荐、跨域推荐、模糊回答、PDF总结），各场景Agent节点独立编排Tool Calling与上下文组装逻辑，通过State Graph管理状态流转与条件分支
- 工具链编排：将RAG检索链封装为LangChain Tool，设计Retrieval Tool、Summary Tool、Recommendation Tool等自定义工具，Agent根据意图自主选择工具组合，而非硬编码调用链路
- 上下文工程：基于ConversationBufferMemory与ChatMessageHistory管理多轮对话上下文，对检索结果进行相关性过滤与窗口裁剪后注入Prompt，结合Streaming实现实时流式响应

#### CE政策解读Agent
- 面向海量政策文件（docx/pdf），设计文档智能解析流程，自动识别并结构化提取省份、年份、月份等元数据；替代了原有人工逐篇查阅政策的方式，使业务人员从"按文件翻找"转变为"按问题检索"
- 以Harness Engineer思想搭建DeepAgent架构，基于LangGraph设计四步Pipeline，每一步精心控制大模型的输入上下文：Query Rewrite（大模型重写用户问题，注入政策领域知识提升召回率）→ Parallel Retrieval（按省份、年份、政策类型等维度并发调用多个检索工具）→ Context Assembly（核心上下文工程环节：对多源检索结果去重、按相关性排序、裁剪至最优窗口长度，拼接结构化Prompt模板）→ Generation（大模型基于精编上下文生成结构化政策总结）
- 开发过程中发现单次检索难以覆盖跨省份、跨年份的复合查询，通过Conditional Edge实现动态流程控制，Agent根据检索质量评分自主决定二次检索或切换检索策略，解决了复合条件下的召回不足问题；支持多轮追问，对话历史经压缩摘要后注入上下文，逐步缩小政策范围

#### MSDP销售赋能Agent
- 通过Prompt Engineering设计结构化提取指令，大模型自动提取文章关键信息并对医学观念进行三级标注（谨慎/中立/倡导），替代了原有人工逐篇阅读标注的方式，为销售拜访医学专家提供精准参考内容
- 同样采用Harness Engineer思想，复用DeepAgent Pipeline架构（Query Rewrite → Parallel Retrieval → Context Assembly → Generation），重点优化Context Assembly环节：将文章标签、观念分类、用户历史偏好等多维信息编排为结构化上下文，驱动大模型生成针对性拜访建议
- Agent通过Function Calling动态调用标签检索、文章检索、观念分析等自定义工具，自主编排调用链路；支持多轮对话持续优化推荐内容

### AI工程化平台（2024.09-2025.03）
**角色：** 大模型全栈开发工程师

#### IResearch AutoML平台
- 改造AutoML服务架构，封装模型动态加载与热切换机制；重新设计Pipeline节点间数据传递方案，开发多因素分析节点与节点分类功能

#### Genlitex TDLC平台
- 主导平台架构设计，基于FastAPI + Keycloak + Celery搭建异步开发框架与用户权限体系；设计Pipeline流程引擎及表结构
- 基于CrewAI多Agent框架开发代码生成系统：Product Manager Agent（需求分析拆解）→ UI Designer Agent（界面布局设计）→ Frontend Developer Agent（Vue3代码生成），三个角色Agent通过Sequential Process协作，实现从自然语言需求到可运行Vue3原型图的全链路自动生成；将原型设计周期从数天缩短至小时级

#### Accelleron Gen-AI平台
- 主导全栈开发（FastAPI + Vue3），集成Azure OpenID SSO单点登录；独立完成从架构设计到前后端开发的全栈交付
- 基于LangGraph开发Human-in-the-Loop Agent：Document Parsing节点（微软PHI模型解析航海日志/轮机日志/加油单PDF）→ Data Extraction节点（结构化字段提取生成JSON）→ Human Review节点（通过LangGraph interrupt机制暂停Agent等待人工确认修改，用户反馈后Agent继续后续流程）；相比纯自动化方案，Human-in-the-Loop机制使数据提取准确率大幅提升，同时保留了人工纠偏能力
- 设计三类自定义Tool：PDF Parsing Tool、Data Extraction Tool、Tekomar API Tool，Agent根据文档类型自主选择工具链；提取数据与Tekomar系统数据自动校验比对，将原本需数小时的人工比对工作缩短至分钟级

### AI工程化平台（2024.06-2024.08）
**角色：** AI工程化开发工程师
- 主导AI工程化平台架构设计与数据建模，设计三层表结构体系，每张表预留JSONB字段以兼容特殊数据结构，确保模型配置的灵活扩展
- 基于FastAPI构建高性能异步API框架，结合SQLAlchemy ORM实现数据库异步操作，通过异步IO（aiohttp/aiofiles）优化文件读写及外部接口调用，确保高并发场景下的系统稳定性
- 联动Dify知识库与聊天API接口，构建智能推荐系统，实现试算场景的快速匹配，使非AI背景的财经人员也能通过数据快速完成场景试算
- 利用Jinja2模板引擎动态生成Kubeflow Pipeline运行脚本，精确控制各节点任务调度逻辑；针对Kubeflow节点独立运行的特性，采用华为S3对象存储作为节点间数据交互中介，确保数据高效传输与一致性
- 采用MLflow记录模型训练过程中的评估指标及模型信息，构建基于FastAPI与Kubeflow的模型预测服务API，支持容器化多实例负载均衡部署

### AutoML平台（2023.09-2024.05）
**角色：** AI工程化开发工程师
- 参与AutoML平台架构设计与优化，协助用户根据各自数据特点和处理需求生成多样化模型并进行精准预测
- 基于Flask框架封装多个数据处理节点的API接口，将模型训练算法封装为HTTP服务，实现灵活调度多种算法以生成不同模型
- 结合Flask和Watchdog技术，构建自动检测并加载最新模型的预测服务API，实现模型热更新，无需手动重启即可上线最新模型
- 设计并实施Airflow工作流编排，保障数据处理与模型训练的顺畅协作；运用MLflow实现训练模型的全面管理与版本追踪
- 采用Nginx Upstream策略部署多个算法服务实例，显著提升模型训练并行效率

### GRB转EFD平台（2022.03-2023.08）
**角色：** 平台设计负责人
- 担任后端技术负责人，基于Go Gin框架搭建开发框架（日志框架、异常处理、Swagger接口文档），设计接口调用流程
- 将原有功能模块（数据加载、过滤、路口更新、区间切割、立体交叉点生产、地物关联、区间关联、应急车道生成、数据存储等9个模块）改造为API接口，实现不同业务功能模块的灵活组合调用，按需编排生成不同的业务数据
- 使用Go geom包封装通用GIS计算方法（线的切割、两条线同向检查、点线面有效性校验等），提升代码可维护性与可扩展性
- 通过Profile性能分析定位耗时瓶颈，针对性优化关键模块，将全上海高速数据处理耗时从42小时降至23小时，效率提升45%
- 使用异步调用方式组合功能模块，进一步提升整体处理效率

### 供应链决策平台（2021.09-2022.02）
**角色：** 后端开发Leader
- 担任后端开发Leader，负责对接需求、任务拆分分配，设计并搭建基于Go Gin的开发框架（日志框架、异常处理、Swagger接口文档）
- 设计角色权限控制流程，精确到每个按钮级别；编写权限拦截中间件，同时记录用户操作日志
- 编写通用工具类（Excel工具、日期工具、类校验工具等），开发角色权限模块、预算模块、计划模块等核心功能
- 收集各部门月度计划（销售、排产、采购、配方、库存、价格、成本），经测算得出最合理计划并下发
- 使用Redis缓存及异步任务提高响应速度，使用Corn定时任务每天凌晨同步SSO用户信息以分配部门权限

### Botzero智能机器人平台（2019.09-2021.08）
**角色：** 后端开发工程师
- 参与智能机器人平台架构设计，引入语料辅助抽取功能（文件上传、解析、标注），可快速提取训练语料；引入语料扩展功能，快速增加训练语料提高模型准确率
- 引入纠错功能和语料查重功能，提高语料质量从而保证模型质量；引入同义词、停用词、特殊词处理，进一步提高模型准确率
- 引入词槽配置，实现用户在一个Bot下的意图跳转及关键信息扣取；引入反馈机制和容错机制，提升Bot智能交互能力
- 以工程化方式替换部分算法模型逻辑（如点击进入Bot），大幅提高响应速度与用户体验；单次模型训练量从2000条问答对提升至10000条
- 引入Elasticsearch优化智能问答机器人检索性能；采用模型热加载技术优化框架架构，提升服务稳定性；使用Nginx为各服务做反向代理

### CEPH监控平台（2018.11-2019.08）
**角色：** 开发组长
- 担任开发小组长，与产品经理对接需求并整理需求文档；负责整个项目表的设计，编写后端Restful API接口文档
- 利用Kafka、Requests和Gevent，将Prometheus的监控数据存入TiDB；由于实时监控需每分钟查询大量指标数据，采用Gevent协程技术解决高频并发查询问题
- 使用Requests技术对InsightFinder（日志和指标分析软件）的数据进行解析和存储
- 针对千万级和百万级数据量的表，建立SQL索引和Redis缓存实现快速查询；选用Redis提高整体响应速度
- 前期代码设计时充分考虑可扩展性，便于后续增加集群、节点、OSD和指标等

## 我的职业发展意向
- 持续深耕Harness Engineer方向，以工程化手段系统性地驾驭大模型，成为AI Agent架构专家
- 探索多Agent协作、自主决策等前沿Agent技术，推动企业级AI应用落地
- 深化上下文工程（Context Engineering）与工具链编排能力，为医药、工业制造等行业构建高价值AI解决方案

## 其他相关信息
- 持有阿里云ACP、华为云HCIP及AWS Certified AI Practitioner认证
- 具备丰富的全球财富500强客户交付经验，擅长跨文化团队协作

## 可选模块
### 认证
- 阿里云ACP（云计算）
- 华为云HCIP
- AWS Certified AI Practitioner

### 工具与平台
- Agent框架：LangGraph（State Graph/Conditional Edge/Human-in-the-Loop）、LangChain（Tool Calling/Memory/Streaming）、CrewAI（Multi-Agent/Sequential Process）
- 大模型与Embedding：Azure OpenAI、通义千问、微软PHI、BGE-Reranker-Large、M3E-Base
- 向量数据库：Milvus（稠密/稀疏/混合检索）、Chroma、Weaviate
- 后端框架：FastAPI、Flask、Django、Gin
- MLOps：MLflow、Airflow、Kubeflow
- 基础设施：Docker、Docker-Compose、K8S、Nginx
- 数据库：PostgreSQL、MySQL、Redis、MongoDB
- 消息队列/搜索：Kafka、RabbitMQ、Elasticsearch、Celery

### 语言能力
- 英语四级

### 联系方式或个人主页
- N/A