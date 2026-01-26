"""Expression generator for symbolic regression.

使用 Vocabulary 中定义的符号集生成随机数学表达式。
"""

import sympy as sp
import numpy as np
from typing import Tuple, List, Dict, Union
from src.model.vocab import Vocabulary


# 运算符元数定义
OPERATOR_ARITY = {
    'add': 2,
    'sub': 2,
    'mul': 2,
    'div': 2,
    'pow': 2,
    'sin': 1,
    'cos': 1,
    'tan': 1,
    'exp': 1,
    'ln': 1,
    'sqrt': 1,
    'arcsin': 1,
    'arccos': 1,
    'arctan': 1,
}

# SymPy 运算符映射
SYMPY_OPERATORS = {
    'add': sp.Add,
    'sub': sp.Add,
    'mul': sp.Mul,
    'div': sp.Mul,
    'pow': sp.Pow,
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'exp': sp.exp,
    'ln': sp.log,
    'sqrt': sp.sqrt,
    'arcsin': sp.asin,
    'arccos': sp.acos,
    'arctan': sp.atan,
}


def generate_expression(
    num_variables: int,
    max_depth: int,
    vocab: Vocabulary,
    rng: np.random.Generator,
) -> sp.Expr:
    """生成随机 SymPy 表达式.

    使用递归下降方法生成表达式树。

    Args:
        num_variables: 变量数量 (x0, x1, ...)
        max_depth: 最大深度
        vocab: 词表对象
        rng: 随机数生成器

    Returns:
        SymPy 表达式
    """
    # 分离一元和二元运算符
    binary_ops = [op for op in vocab.OPERATORS if OPERATOR_ARITY[op] == 2]
    unary_ops = vocab.FUNCTIONS

    def _generate(depth: int) -> sp.Expr:
        # 到达最大深度或随机停止时，返回变量
        if depth >= max_depth or (depth > 0 and rng.random() < 0.3):
            var_idx = rng.integers(0, num_variables)
            return sp.Symbol(f'x{var_idx}')

        # 随机选择运算符类型
        if rng.random() < 0.5 and unary_ops:
            # 一元运算符
            op = rng.choice(unary_ops)
            arg = _generate(depth + 1)
            return SYMPY_OPERATORS[op](arg)
        else:
            # 二元运算符
            op = rng.choice(binary_ops)
            left = _generate(depth + 1)
            right = _generate(depth + 1)

            if op == 'sub':
                return left - right
            elif op == 'div':
                return left / right
            else:
                return SYMPY_OPERATORS[op](left, right)

    return _generate(0)


def sample_points(
    n_points: int,
    x_range: Tuple[float, float],
    num_variables: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """采样输入点.

    Args:
        n_points: 采样点数量
        x_range: 采样范围
        num_variables: 变量数量
        rng: 随机数生成器

    Returns:
        x_values: (n_points, num_variables) 输入特征
    """
    x_values = rng.uniform(x_range[0], x_range[1], size=(n_points, num_variables))
    return x_values


def evaluate_expression(
    expr: sp.Expr,
    x_values: np.ndarray,
) -> Union[np.ndarray, bool]:
    """计算表达式在采样点的值.

    Args:
        expr: SymPy 表达式
        x_values: (n_points, num_variables) 输入特征

    Returns:
        y_target: (n_points,) 目标值
        False: 如果存在复数或异常值
    """
    # 检测表达式中的无效常量（ComplexInfinity, NaN, Infinity等）
    for atom in expr.atoms():
        if atom in (sp.zoo, sp.nan, sp.oo, sp.S.NegativeInfinity):
            return False

    # 检测 AccumulationBounds（使用类型检查）
    from sympy.calculus.accumulationbounds import AccumulationBounds
    if any(isinstance(sub_expr, AccumulationBounds) for sub_expr in sp.postorder_traversal(expr)):
        return False

    _, num_variables = x_values.shape

    # 创建符号变量
    symbols = [sp.Symbol(f'x{i}') for i in range(num_variables)]

    # 将表达式转换为可调用函数
    f = sp.lambdify(symbols, expr, 'numpy')

    # 计算目标值（抑制数学运算警告，出现 NaN/Inf 会返回 False）
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        y_target = f(*[x_values[:, i] for i in range(num_variables)])

    # 检测复数（在转换前）
    if np.iscomplexobj(y_target):
        return False

    # 确保输出形状正确（常量表达式会返回标量）
    y_target = np.asarray(y_target, dtype=np.float64)
    if y_target.ndim == 0:
        y_target = np.full(x_values.shape[0], y_target)

    # 检测 NaN 和 Inf
    if np.any(np.isnan(y_target)) or np.any(np.isinf(y_target)):
        return False

    # 检测绝对值过大（> 100）
    if np.any(np.abs(y_target) > 10):
        return False

    return y_target


def generate_expression_sample(
    num_variables: int = 3,
    max_depth: int = 3,
    n_points: int = 500,
    x_range: Tuple[float, float] = (-10, 10),
    vocab: Vocabulary = None,
    rng: np.random.Generator = None,
) -> Union[Tuple[sp.Expr, sp.Expr, np.ndarray, np.ndarray], bool]:
    """生成 base 表达式、target 表达式、采样点并计算目标值.

    Returns:
        (base_expr, target_expr, x_values, y_target): 成功时返回四元组
        False: 失败时返回 False
    """
    if vocab is None:
        vocab = Vocabulary(num_variables=num_variables)

    if rng is None:
        rng = np.random.default_rng()

    # 1. 生成 target 表达式
    target_expr = generate_expression(num_variables, max_depth, vocab, rng)

    # 2. 生成 base 表达式（target 的简化版本）
    simplify_ratio = rng.uniform(0.3, 0.7)
    base_expr = simplify_sympy_expr(target_expr, rng, simplify_ratio)

    # 3. 采样输入点
    x_values = sample_points(n_points, x_range, num_variables, rng)

    # 4. 计算 target 表达式在采样点的值
    y_target = evaluate_expression(target_expr, x_values)

    if y_target is False:
        return False

    # 5. 验证 base_expr 的 tokens 是否在词表中
    base_tokens = expression_to_tokens(base_expr)
    try:
        [vocab.token_to_id(token) for token in base_tokens]
    except KeyError:
        return False

    return base_expr, target_expr, x_values, y_target


def expression_to_tokens(expr: sp.Expr) -> List[str]:
    """将 SymPy 表达式转换为前缀 token 序列.

    Args:
        expr: SymPy 表达式

    Returns:
        tokens: 前缀 token 列表
    """
    tokens = []

    def _preorder(node):
        if node.is_Atom:
            if isinstance(node, sp.Symbol):
                tokens.append(str(node))
            elif isinstance(node, sp.Number):
                tokens.append('constant')
            elif node in (sp.pi, sp.E, sp.GoldenRatio):
                tokens.append('constant')
            else:
                tokens.append(str(node))
        else:
            op_name = _get_operation_name(node)
            tokens.append(op_name)
            for arg in node.args:
                _preorder(arg)

    def _get_operation_name(node):
        if node.is_Add:
            return 'add'
        elif node.is_Mul:
            return 'mul'
        elif node.is_Pow:
            return 'pow'
        elif node.func == sp.sin:
            return 'sin'
        elif node.func == sp.cos:
            return 'cos'
        elif node.func == sp.tan:
            return 'tan'
        elif node.func == sp.exp:
            return 'exp'
        elif node.func == sp.log:
            return 'ln'
        elif node.func == sp.sqrt:
            return 'sqrt'
        elif node.func == sp.asin:
            return 'arcsin'
        elif node.func == sp.acos:
            return 'arccos'
        elif node.func == sp.atan:
            return 'arctan'
        elif isinstance(node, sp.NegativeOne):
            return '-1'
        else:
            return str(node.func)

    _preorder(expr)
    return tokens


def simplify_sympy_expr(expr: sp.Expr, rng: np.random.Generator, simplify_ratio: float = 0.5) -> sp.Expr:
    """简化 SymPy 表达式

    策略：随机选择非叶子子表达式，用其子表达式之一替换
    """
    if expr.is_Atom:
        return expr

    # 收集可简化的子表达式
    candidates = []
    for sub_expr in sp.postorder_traversal(expr):
        if sub_expr == expr:
            continue
        if not sub_expr.is_Atom and len(sub_expr.args) > 1:
            candidates.append(sub_expr)

    if not candidates:
        return expr

    # 随机选择要简化的节点
    num_to_simplify = max(1, int(len(candidates) * simplify_ratio))
    nodes_to_simplify = rng.choice(candidates, size=min(num_to_simplify, len(candidates)), replace=False)

    simplified = expr
    for node in nodes_to_simplify:
        # 用随机子节点替换当前节点
        replacement = rng.choice(list(node.args))
        try:
            simplified = simplified.subs(node, replacement)

            # 检查是否引入了虚数单位
            if sp.I in simplified.atoms():
                # 恢复为未简化的表达式
                simplified = expr
                break

            # 检查是否产生了 AccumulationBounds 或其他无效类型
            for atom in simplified.atoms():
                atom_type = type(atom).__name__
                if atom_type == 'AccumulationBounds':
                    simplified = expr
                    break
        except (ValueError, TypeError):
            simplified = expr
            break

    return simplified
