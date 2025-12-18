import os
import shutil
import argparse
import chromadb
import torch
import clip
import re
from PIL import Image
#from pypdf import PdfReader
import pdfplumber
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

model_cache_dir="./model_cache"
db_path="./db"
class Agent:
    def __init__(self, db_path="./db"):
        print("正在初始化本地 AI 智能文献与图像管理助手...")
        os.makedirs(model_cache_dir, exist_ok=True)
        #初始化向量数据库
        self.chroma_client=chromadb.PersistentClient(path=db_path)
        self.img=self.chroma_client.get_or_create_collection(name="images",
                                                             metadata={"hnsw:space": "cosine"} )
        self.doc=self.chroma_client.get_or_create_collection(name="documents",
                                                             metadata={"hnsw:space": "cosine"} )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #加载CLIP模型
        print("正在加载clip模型...")
        self.model, self.preprocess = clip.load(
            "ViT-B/32",
              device=self.device,
              download_root=model_cache_dir)
        #支持中文的CLIP模型
        self.multilingual_clip = SentenceTransformer(
            'sentence-transformers/clip-ViT-B-32-multilingual-v1', 
            device=self.device,
            cache_folder=model_cache_dir)
        #加载文本嵌入模型(支持中文)
        print("正在加载文本嵌入模型...")
        self.text_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2', 
            device=self.device,
            cache_folder=model_cache_dir 
        )        
        #加载文本分类模型
        print("正在加载文本分类模型...")
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli", 
            device=0 if torch.cuda.is_available() else -1,
            cache_dir=model_cache_dir
        )
        print("模型加载完毕！")

    #添加图片到数据库
    def add_image(self, path):
        image_files=[]
        valid_extensions=('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        image_files.append(os.path.join(root, file))
        elif os.path.isfile(path) and path.lower().endswith(valid_extensions):
            image_files=[path]
        else:
            print(f"无效路径或非图片/文件夹: {path}")
            return
        count=0
        for idx,image_path in enumerate(image_files):
            try:
                #处理并添加图像到数据库
                pil_image = Image.open(image_path).convert("RGB")
                image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    image_embedding = image_features.cpu().numpy().flatten().tolist()
                imageid=os.path.basename(image_path)
                self.img.add(
                    ids=[imageid],
                    embeddings=[image_embedding],
                    metadatas=[{"path":image_path}]
                )
                count+=1
                if (idx+1)%10==0:
                    print(f"已添加 {idx+1} 张图片")
            except Exception as e:
                print(f"添加图片失败: {image_path}, 错误: {e}")
        print(f"添加了{count}张图片到数据库")
    #检索图片
    def search_img(self,query,topk=3):
        print(f"正在搜索图片:{query}")
        
        query_embedding = self.multilingual_clip.encode(
            query, 
            convert_to_tensor=False, 
            normalize_embeddings=True
        ).tolist()
        #text_token=clip.tokenize([query]).to(self.device)
        #with torch.no_grad():
        #    query_features=self.model.encode_text(text_token)
        #    query_features /= query_features.norm(dim=-1, keepdim=True)
        #    query_embedding = query_features.cpu().numpy().flatten().tolist()
        results=self.img.query(query_embeddings=[query_embedding],n_results=topk)
        if not results['ids'][0]:
            print("没有找到相关图片")
            return
        print(f"找到 {len(results['ids'][0])} 张相关图片:")
        for i,filename in enumerate(results['ids'][0]):
            metadata=results['metadatas'][0][i]
            score=results['distances'][0][i]
            print(f"图片: {filename}, 路径: {metadata['path']}, 相似度: {score:.4f}")
    #提取摘要
    def extract_abstract(self, path):
        try:
            #reader=PdfReader(path)
            text=""
            with pdfplumber.open(path) as pdf:
                #只读前2页
                pages = pdf.pages[:2]
                for page in pages:
                    page_text =page.extract_text(x_tolerance=1)
                    if page_text:
                        text += page_text + "\n"
                text = text.replace('\n', ' ')

            #正则式匹配
            pattern = r"(?i)\babstract\b[:\.]?\s*(.*?)\s*(?:1\.?|I\.?)?\s*introduction"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                abstract =match.group(1).strip()
                abstract =abstract.replace('-\n', '').replace('\n', ' ')
                abstract =re.sub(r'\b\d{1,2}\b\s+',' ', abstract) #去掉连续的单个数字
                return abstract
            #关键词截取
            lower_text = text.lower()
            abs_start = lower_text.find("abstract")
            if abs_start != -1:
                abstract=text[abs_start+8:abs_start+500]
                abstract =abstract.strip().replace('-\n', '').strip()
                abstract =re.sub(r'\b\d{1,2}\b\s+',' ', abstract) # 去掉连续的单个数字
                return abstract
            #默认截取300-1500字符
            return text[300:1500].replace('\n', ' ').strip()
        except Exception as e:
            print(f"提取摘要失败: {path}, 错误: {e}")
            return ""
    #文本切片
    def _chunk_text(self, text, chunk_size=500, overlap=50):
        if not text: return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            # 移动窗口
            start += (chunk_size - overlap)
        return chunks
    #添加文档到数据库
    def add_document(self, path):
        pdf_files=[]
        if os.path.isdir(path):
            print(f"正在索引文件夹: {path}")
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
        elif os.path.isfile(path) and path.lower().endswith('.pdf'):
            pdf_files=[path]
        else:
            print(f"无效路径或非PDF文件/文件夹: {path}")
            return
        print(f"找到 {len(pdf_files)} 个PDF文件，开始索引...")
        count=0
        for idx,pdf_path in enumerate(pdf_files):
            try:
                filename=os.path.basename(pdf_path)
                with pdfplumber.open(pdf_path) as pdf:
                    num_pages = len(pdf.pages)
                    chunks = []
                    embedding=[]
                    ids=[]
                    metadata=[]
                    #遍历每页
                    for page_idx,page in enumerate(pdf.pages):
                        text=page.extract_text(x_tolerance=1)
                        if not text:
                            continue
                        text=text.replace('\n', ' ')
                        #提取chunk
                        page_chunks=self._chunk_text(text,chunk_size=500,overlap=50)
                        #遍历chunk
                        for chunk_idx,chunk in enumerate(page_chunks):
                            if len(chunk)<20: continue
                            chunks.append(chunk)

                            uid=f"{filename}_p{page_idx+1}_c{chunk_idx}"
                            ids.append(uid)
                            #构造metadata
                            metadata.append(
                                {"path":pdf_path,
                                 "filename":filename,
                                 "page":page_idx+1,}
                            )
                    if not chunks:
                        print(f"无法提取文本:{os.path.basename(pdf_path)}，跳过索引")
                        return
                    print(f"   -> 正在计算 {len(chunks)} 个片段的向量...")
                    embedding=self.text_model.encode(
                                chunks,
                                convert_to_tensor=False,
                                normalize_embeddings=True,
                                batch_size=32#批处理
                    ).tolist()
                    #存储
                    self.doc.add(
                                ids=ids,
                                embeddings=embedding,
                                documents=chunks,
                                metadatas=metadata)
                    print(f"成功索引 {num_pages} 页，共 {len(chunks)} 个片段。")
                    count+=1
                    if (idx+1)%5==0:
                        print(f"已索引 {idx+1} 个文档")

            except Exception as e:
                print(f"索引文档失败: {pdf_path}, 错误: {e}")
        print(f"成功索引了 {count} 个文档到数据库")
    #搜索文档
    def search_document(self,query,topk=3):
        print(f"正在搜索文档:{query}")
        query_embedding=self.text_model.encode(
            query,
            convert_to_tensor=False,
            normalize_embeddings=True
        ).tolist()
        results=self.doc.query(query_embeddings=[query_embedding],n_results=topk)
        if not results['ids'][0]:
            print("没有找到相关文档")
            return
        print(f"找到 {len(results['ids'][0])} 个相关文档:")


        for i,filename in enumerate(results['ids'][0]):
            
            metadata=results['metadatas'][0][i]
            content = results['documents'][0][i]
            score=1-results['distances'][0][i]
            filename=metadata.get("filename", "Unknown")
            page_num=metadata.get("page", "N/A")
            print(f"来源: {filename} (第 {page_num} 页)")
            print(f"置信度: {score:.4f}")
            print(f"片段内容: \"...{content}...\"")
    # 自动分类
    def org_docunments(self,folder,categoories):
        categoories= [cat.strip() for cat in categoories.split(',')]
        #categoories=["Reinforcement Learning, RL, Q-Learning, Policy Gradient, Reward, Agent",
        #"Computer Vision, CV, Image Classification, Object Detection, Segmentation, ImageNet, CNN",  # <--- 加入具体任务名
        #"Large Language Model, LLM, NLP, Transformer, Text Generation, BERT, GPT"]
        if not categoories:
            print("请提供分类主题")
            return
        files=[]
        if os.path.isfile(folder):
            files.append(folder)
            root_dir=os.path.dirname(folder)
        elif os.path.isdir(folder):
            for root, dirs, fs in os.walk(folder):
                for file in fs:
                    if file.lower().endswith('.pdf'):
                        files.append(os.path.join(root, file))
        else:
            print(f"路径无效")
            return
        print(f"需整理文件数:{len(files)}")
        for file in files:
            try:
                title = os.path.basename(file).replace('.pdf', '').replace('_', ' ')
                filename=os.path.basename(file)
                current_dir = os.path.dirname(file)
                abstract = self.extract_abstract(file)
                #print(f"\n--- [DEBUG] 正在分析文件: {filename} ---")
                #print(f"提取到的文本前500字: >>> {abstract[:500]} <<<\n")
                if len(abstract)<50:
                    print(f"无法提取摘要或摘要过短:{filename}，跳过整理")
                    continue
                combined_text = f"{title}. {abstract}"#标题+摘要
                result = self.classifier(combined_text, categoories, multi_label=False)
                best_cat = result['labels'][0] 
                score = result['scores'][0]    
        
                print(f"{title} ->  {best_cat} (置信度: {score:.4f})")

                print(f"文件:{filename} 分类到 '{best_cat}' (置信度:{score:.4f})")
                
                cat_dir=os.path.join(current_dir,best_cat)
                os.makedirs(cat_dir,exist_ok=True)

                cat_path=os.path.join(cat_dir,filename)
                if os.path.abspath(file) == os.path.abspath(cat_path):
                    print(f"   已在目标文件夹中，跳过。")
                    continue
                shutil.move(file,cat_path)
            except Exception as e:
                print(f"整理文件失败:{filename},错误:{e}")
                

            


def main():
    parser=argparse.ArgumentParser(description="本地 AI 智能文献与图像管理助手")
    subparsers=parser.add_subparsers(dest='command',help="可用命令")
    #添加图片
    add_img=subparsers.add_parser('add_image',help="添加图片到数据库")
    add_img.add_argument('path',type=str,help="图片文件或文件夹路径")
    #搜图片
    search_img=subparsers.add_parser('search_image',help="以文搜图")
    search_img.add_argument('query',type=str,help="搜索文本")
    #添加文档
    idx_doc = subparsers.add_parser('add_doc', help="添加PDF文档")
    idx_doc.add_argument('path', type=str, help="PDF路径")

    #搜文档
    search_doc = subparsers.add_parser('search_doc', help="语义搜索文档")
    search_doc.add_argument('query', type=str, help="搜索关键词")

    #整理文档
    sort = subparsers.add_parser('sort', help="分类整理 (支持单文件或文件夹)")
    sort.add_argument('target', type=str, help="待处理的文件或文件夹路径")
    sort.add_argument('--topics', type=str, required=True, help="类别,逗号分隔 (如: 'CV,NLP,金融')")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    agent = Agent()

    #命令
    if args.command == 'add_image':
        agent.add_image(args.path) 
    elif args.command == 'search_image':
        agent.search_img(args.query)
    elif args.command == 'add_doc':
        agent.add_document(args.path)
    elif args.command == 'search_doc':
        agent.search_document(args.query)
    elif args.command == 'sort':
        agent.org_docunments(args.target, args.topics)


if __name__ == "__main__":
    main()

