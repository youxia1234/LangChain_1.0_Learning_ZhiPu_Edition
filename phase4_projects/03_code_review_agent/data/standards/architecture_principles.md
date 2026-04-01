# 软件架构设计原则

## SOLID 原则

### 单一职责原则 (SRP)

一个类或模块应该只有一个引起它变化的原因。

```python
# 违反 SRP - 一个类做太多事
class UserManagement:
    def create_user(self, data):
        # 验证数据
        if not data.get("email"):
            raise ValueError("邮箱不能为空")
        # 保存到数据库
        db.save("users", data)
        # 发送欢迎邮件
        send_email(data["email"], "欢迎注册")
        # 记录日志
        logger.info(f"新用户注册: {data['email']}")

# 遵循 SRP - 拆分职责
class UserValidator:
    def validate(self, data):
        if not data.get("email"):
            raise ValueError("邮箱不能为空")
        return data

class UserRepository:
    def save(self, data):
        return db.save("users", data)

class EmailService:
    def send_welcome(self, email):
        send_email(email, "欢迎注册")

class UserService:
    """编排层 - 协调各组件"""
    def __init__(self, validator, repo, email_service):
        self.validator = validator
        self.repo = repo
        self.email_service = email_service

    def create_user(self, data):
        validated = self.validator.validate(data)
        user = self.repo.save(validated)
        self.email_service.send_welcome(user["email"])
        return user
```

### 开闭原则 (OCP)

软件实体应对扩展开放，对修改关闭。

```python
# 违反 OCP - 每增加一种折扣都要修改函数
def calculate_price(order, discount_type):
    if discount_type == "percentage":
        return order.total * (1 - order.discount_rate)
    elif discount_type == "fixed":
        return order.total - order.discount_amount
    elif discount_type == "buy_one_get_one":
        return order.total * 0.5
    # 新增折扣类型需要修改这里

# 遵循 OCP - 使用策略模式
from abc import ABC, abstractmethod

class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, order) -> float:
        pass

class PercentageDiscount(DiscountStrategy):
    def calculate(self, order):
        return order.total * (1 - order.discount_rate)

class FixedDiscount(DiscountStrategy):
    def calculate(self, order):
        return order.total - order.discount_amount

class BuyOneGetOneDiscount(DiscountStrategy):
    def calculate(self, order):
        return order.total * 0.5

# 新增折扣只需添加新类，不需要修改已有代码
class VIPDiscount(DiscountStrategy):
    def calculate(self, order):
        return order.total * 0.7
```

### 里氏替换原则 (LSP)

子类应该能替换其基类而不破坏程序正确性。

```python
# 违反 LSP - 子类改变了基类行为
class Bird:
    def fly(self):
        return "飞翔"

class Penguin(Bird):
    def fly(self):
        raise NotImplementedError("企鹅不会飞")  # 破坏了约定

# 遵循 LSP - 合理的继承层次
class Bird:
    def move(self):
        return "移动"

class FlyingBird(Bird):
    def fly(self):
        return "飞翔"

class FlightlessBird(Bird):
    def swim(self):
        return "游泳"

class Eagle(FlyingBird):
    pass

class Penguin(FlightlessBird):
    pass
```

### 接口隔离原则 (ISP)

不应强迫类实现它不需要的接口。

```python
# 违反 ISP - 胖接口
class Worker(ABC):
    @abstractmethod
    def work(self): pass

    @abstractmethod
    def eat(self): pass

    @abstractmethod
    def sleep(self): pass

class Robot(Worker):
    def work(self):
        return "工作中"
    def eat(self):
        pass  # 机器人不需要吃饭
    def sleep(self):
        pass  # 机器人不需要睡觉

# 遵循 ISP - 拆分接口
class Workable(ABC):
    @abstractmethod
    def work(self): pass

class Eatable(ABC):
    @abstractmethod
    def eat(self): pass

class Human(Workable, Eatable):
    def work(self): return "工作中"
    def eat(self): return "吃饭中"

class Robot(Workable):
    def work(self): return "工作中"
```

### 依赖倒置原则 (DIP)

高层模块不应依赖低层模块，两者都应依赖抽象。

```python
# 违反 DIP - 直接依赖具体实现
class OrderService:
    def __init__(self):
        self.db = PostgresDatabase()      # 硬编码依赖
        self.notifier = EmailNotifier()    # 硬编码依赖

    def create_order(self, data):
        order = self.db.save("orders", data)
        self.notifier.send(order)
        return order

# 遵循 DIP - 依赖抽象（依赖注入）
class Database(ABC):
    @abstractmethod
    def save(self, table, data): pass

class Notifier(ABC):
    @abstractmethod
    def send(self, data): pass

class OrderService:
    def __init__(self, db: Database, notifier: Notifier):
        self.db = db
        self.notifier = notifier

    def create_order(self, data):
        order = self.db.save("orders", data)
        self.notifier.send(order)
        return order
```

## 常用设计模式

### 工厂模式

将对象创建逻辑集中管理。

```python
# 不使用工厂 - 分散的创建逻辑
if processor_type == "image":
    processor = ImageProcessor(config)
elif processor_type == "video":
    processor = VideoProcessor(config)
elif processor_type == "audio":
    processor = AudioProcessor(config)

# 使用工厂
class ProcessorFactory:
    _processors = {
        "image": ImageProcessor,
        "video": VideoProcessor,
        "audio": AudioProcessor,
    }

    @classmethod
    def create(cls, processor_type: str, config: dict):
        processor_class = cls._processors.get(processor_type)
        if not processor_class:
            raise ValueError(f"未知处理器类型: {processor_type}")
        return processor_class(config)
```

### 观察者模式

解耦事件产生者和消费者。

```python
# 不使用观察者 - 紧耦合
class OrderService:
    def create_order(self, data):
        order = save_order(data)
        email_service.send_confirmation(order)
        inventory_service.reserve_items(order)
        analytics_service.track_order(order)
        # 每增加一个消费者都要修改这里

# 使用观察者
class EventBus:
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event_type, handler):
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event_type, data):
        for handler in self._subscribers.get(event_type, []):
            handler(data)

bus = EventBus()
bus.subscribe("order_created", email_service.send_confirmation)
bus.subscribe("order_created", inventory_service.reserve_items)
bus.subscribe("order_created", analytics_service.track_order)
```

## 错误处理策略

### 自定义异常层级

```python
# 错误 - 使用通用 Exception
def process(data):
    if not data:
        raise Exception("数据为空")
    if len(data) > 100:
        raise Exception("数据过多")

# 正确 - 自定义异常层级
class AppError(Exception):
    """应用基础异常"""
    pass

class ValidationError(AppError):
    """数据验证错误"""
    pass

class ResourceNotFoundError(AppError):
    """资源未找到"""
    pass

class BusinessRuleError(AppError):
    """业务规则违反"""
    pass

def process(data):
    if not data:
        raise ValidationError("数据不能为空")
    if len(data) > 100:
        raise BusinessRuleError("单次处理数据不能超过100条")
```

### 避免 bare except

```python
# 错误 - 吞掉所有异常
try:
    result = do_something()
except:
    pass  # 隐藏了所有错误

# 错误 - 过于宽泛
try:
    result = do_something()
except Exception:
    pass

# 正确 - 捕获具体异常
try:
    result = do_something()
except (ValueError, KeyError) as e:
    logger.error(f"数据处理失败: {e}")
    raise ValidationError(str(e))
```

## 可测试性设计

### 依赖注入

```python
# 难以测试 - 硬编码依赖
class UserService:
    def get_user(self, user_id):
        response = requests.get(f"http://api/users/{user_id}")  # 硬编码HTTP调用
        return response.json()

# 易于测试 - 依赖注入
class UserService:
    def __init__(self, api_client):
        self.api_client = api_client

    def get_user(self, user_id):
        return self.api_client.get(f"/users/{user_id}")

# 测试时可以注入 mock
def test_get_user():
    mock_client = Mock()
    mock_client.get.return_value = {"id": 1, "name": "Alice"}
    service = UserService(mock_client)
    assert service.get_user(1)["name"] == "Alice"
```

### 避免硬编码时间

```python
# 难以测试 - 硬编码当前时间
def is_expired(expiry_date):
    return expiry_date < datetime.now()

# 易于测试 - 可注入时间源
def is_expired(expiry_date, now=None):
    now = now or datetime.now()
    return expiry_date < now

# 测试
def test_is_expired():
    assert is_expired(
        datetime(2024, 1, 1),
        now=datetime(2024, 6, 1)
    ) == True
```

## 模块化设计原则

### 分层架构

```
controller/  → 接收请求、参数验证、响应格式化
service/     → 业务逻辑编排
repository/  → 数据访问
model/       → 数据结构定义
```

```python
# Controller 层 - 只负责HTTP交互
@app.post("/api/users")
def create_user(request: CreateUserRequest):
    validated = validate_request(request)
    user = user_service.create(validated)
    return UserResponse.from_model(user)

# Service 层 - 业务逻辑
class UserService:
    def create(self, data):
        self.validator.validate(data)
        user = self.repo.save(data)
        self.event_bus.publish("user_created", user)
        return user

# Repository 层 - 数据访问
class UserRepository:
    def save(self, data):
        return self.db.insert("users", data)
```
