import pandas as pd
from gensim.models.word2vec import Word2Vec

word2vec = Word2Vec.load("./train/embedding/node_w2v_128").wv
ast = pd.read_pickle('./data_all_blocks.pkl')
code_ast = ast[ast['id1'] == 74].iloc[0]['code_x']


def getelement(aa):
    for elem in aa:
        if type(elem) == type([]):
            for element in getelement(elem):
                yield element
        else:
            yield elem


for elem in getelement(code_ast):
    try:
        print(elem, word2vec.index2word[elem])
    except:
        print(2957)

"""
    static void copy(String src, String dest) throws IOException {
        File ifp = new File(src);
        
        
        FileInputStream fis = new FileInputStream(ifp);
        
        fis.close();
    }
"""

a = ['MethodDeclaration', 'Modifier', 'static', 'copy', 'FormalParameter', 'ReferenceType', 'String', 'src',
     'FormalParameter', 'ReferenceType', 'String', 'dest', 'IOException', 'LocalVariableDeclaration', 'ReferenceType',
     'File', 'VariableDeclarator', 'ifp', 'ClassCreator', 'ReferenceType', 'File', 'MemberReference', 'src',
     'LocalVariableDeclaration', 'ReferenceType', 'File', 'VariableDeclarator', 'ofp', 'ClassCreator', 'ReferenceType',
     'File', 'MemberReference', 'dest', 'IfStatement', 'BinaryOperation', '==', 'MethodInvocation', 'ifp', 'exists',
     'Literal', 'false', 'BlockStatement', 'ThrowStatement', 'ClassCreator', 'ReferenceType', 'IOException',
     'BinaryOperation', '+', 'BinaryOperation', '+', 'Literal', '""file \'""', 'MemberReference', 'src', 'Literal',
     '""\' does not exist""', 'End', 'LocalVariableDeclaration', 'ReferenceType', 'FileInputStream',
     'VariableDeclarator', 'fis', 'ClassCreator', 'ReferenceType', 'FileInputStream', 'MemberReference', 'ifp',
     'LocalVariableDeclaration', 'ReferenceType', 'FileOutputStream', 'VariableDeclarator', 'fos', 'ClassCreator',
     'ReferenceType', 'FileOutputStream', 'MemberReference', 'ofp', 'LocalVariableDeclaration', 'BasicType', 'byte',
     'VariableDeclarator', 'b', 'ArrayCreator', 'BasicType', 'byte', 'Literal', '1024', 'LocalVariableDeclaration',
     'BasicType', 'int', 'VariableDeclarator', 'readBytes', 'WhileStatement', 'BinaryOperation', '>', 'Assignment',
     'MemberReference', 'readBytes', 'MethodInvocation', 'fis', 'MemberReference', 'b', 'read', '=', 'Literal', '0',
     'StatementExpression', 'MethodInvocation', 'fos', 'MemberReference', 'b', 'Literal', '0', 'MemberReference',
     'readBytes', 'write', 'End', 'StatementExpression', 'MethodInvocation', 'fis', 'close', 'StatementExpression',
     'MethodInvocation', 'fos', 'close']
