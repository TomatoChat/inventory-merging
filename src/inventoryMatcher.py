import glob
import logging
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def checkDuplicatesColumnDifferences(df: pd.DataFrame, columnToCheck: str, columnDuplicated: str = 'Item') -> tuple[int, list[str]]:
    """
    Check for duplicates with different values in specified column.
    
    Args:
        df (pd.DataFrame): DataFrame to check for duplicates
        columnToCheck (str): Column name to check for different values among duplicates
        columnDuplicated (str, optional): Column name used to identify duplicates. Defaults to 'Item'.
    
    Returns:
        tuple: (count, item_list) - Number of items with different values and list of item names
    """
    sortedDuplicates = df.loc[df.duplicated(subset=[columnDuplicated], keep=False)].sort_values('Item')
    differentValues = []

    for duplicate in sortedDuplicates[columnDuplicated].unique():
        duplicateData = sortedDuplicates.loc[sortedDuplicates[columnDuplicated] == duplicate]
        uniqueValues = duplicateData[columnToCheck].nunique()

        if uniqueValues > 1:
            differentValues.append(duplicate)
            
    logging.info(f'Duplicates with different "{columnToCheck}" #{len(differentValues)}')
    return len(differentValues), differentValues


def mergeDuplicates(df: pd.DataFrame, columnsToMerge: dict[str, str], columnDuplicated: list[str] = ['Item']) -> pd.DataFrame:
    """
    Merge duplicate items by aggregating specified columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing items with potential duplicates
        columnsToMerge (dict[str, str]): Dictionary mapping column names to aggregation functions
        columnDuplicated (list[str], optional): Column(s) used to identify duplicates. Defaults to ['Item'].
    
    Returns:
        pd.DataFrame: DataFrame with duplicates merged according to specified aggregation rules
    """
    logging.info(f'Duplicate item #{df.duplicated(subset=columnDuplicated).sum()}')
    mergedDuplicates = df.groupby(columnDuplicated).agg(columnsToMerge).reset_index()
    df = pd.merge(df.drop_duplicates(subset=columnDuplicated, keep='first').drop(columns=list(columnsToMerge.keys())), mergedDuplicates,
                  on=columnDuplicated,
                  how='left')
    
    logging.info(f'Duplicate item #{df.duplicated(subset=columnDuplicated).sum()}')
    return df





def addEmbeddingsToInventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """
    Add embeddings to inventory DataFrame with category-enhanced item names.
    
    Args:
        inventory (pd.DataFrame): DataFrame containing inventory items with 'Item' and 'Category' columns
    
    Returns:
        pd.DataFrame: Original DataFrame with additional 'EnhancedItem' and 'Embeddings' columns
    """
    inventory = inventory.loc[(inventory['Item'].notna()) & (inventory['Item'].str.strip() != '')].reset_index(drop=True)
    
    # Create category-enhanced item names for better matching
    enhancedItemNames = []

    for _, row in inventory.iterrows():
        itemName = row['Item']
        category = row.get('Category', '')

        if category and pd.notna(category):
            enhancedName = f"{category} {itemName}"
        else:
            enhancedName = itemName
        
        enhancedItemNames.append(enhancedName)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(enhancedItemNames).tolist()
    
    embeddingsDF = pd.DataFrame({
        'Item': inventory['Item'].tolist(),
        'EnhancedItem': enhancedItemNames,
        'Embeddings': embeddings
    })
    
    inventory = pd.merge(inventory, embeddingsDF, on=['Item'], how='left')
    
    return inventory


def getTopCandidatesTFIDF(sendingInventory: pd.DataFrame, receivingInventory: pd.DataFrame, topK: int = 50) -> dict:
    """
    For each item in sending inventory, find the topK most similar items in receiving inventory using TF-IDF.
        
    Args:
        sendingInventory (pd.DataFrame): DataFrame containing items to be matched. Must have 'Item' column.
        receivingInventory (pd.DataFrame): DataFrame containing potential matches. Must have 'Item' column.
        topK (int, optional): Number of top similar items to return for each sending item. Defaults to 50.
    
    Returns:
        dict: Dictionary mapping sending item indices to lists of receiving item indices.
    """
    sendingInventoryItems = sendingInventory['EnhancedItem'].tolist()
    receivingInventoryItems = receivingInventory['EnhancedItem'].tolist()
    
    logging.info(f'Creating TF-IDF matrix for all items...')
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    allItems = sendingInventoryItems + receivingInventoryItems
    matrixTFIDF = vectorizer.fit_transform(allItems)
    logging.info(f'Created TF-IDF matrix for all items...')

    sendingInventoryTFIDF = matrixTFIDF[:len(sendingInventoryItems)]
    receivingInventoryTFIDF = matrixTFIDF[len(sendingInventoryItems):]
    similarityMatrixTFIDF = cosine_similarity(sendingInventoryTFIDF, receivingInventoryTFIDF)
    candidatesSimilarity = {}
    
    for i, sendingInventoryIndex in enumerate(sendingInventory.index):
        similarities = similarityMatrixTFIDF[i]
        topCandidates = np.argsort(similarities)[::-1][:topK]
        receivingInventoryItemsIndices = receivingInventory.index[topCandidates].tolist()
        candidatesSimilarity[sendingInventoryIndex] = receivingInventoryItemsIndices

        logging.info(f'"{sendingInventory.loc[sendingInventoryIndex, "Item"]}" -> best match: "{receivingInventory.loc[receivingInventoryItemsIndices[0], "Item"]}" and {len(receivingInventoryItemsIndices) - 1} others')

    return candidatesSimilarity


def calculateEmbeddingDistancesForCandidates(sendingInventory: pd.DataFrame, receivingInventory: pd.DataFrame, candidatesDict: dict) -> dict:
    """
    Calculate embedding distances between sending items and their top TF-IDF candidates.
    
    Args:
        sendingInventory (pd.DataFrame): DataFrame containing sending items with embeddings
        receivingInventory (pd.DataFrame): DataFrame containing receiving items with embeddings  
        candidatesDict (dict): Dictionary mapping sending item indices to lists of receiving item indices
    
    Returns:
        dict: Dictionary mapping sending item indices to (receiving_indices, distances) tuples
    """
    results = {}
    
    for sendingIdx, receivingCandidates in candidatesDict.items():
        # Get the sending item embedding
        sendingItem = sendingInventory.loc[sendingIdx]
        sendingEmbedding = np.array(sendingItem['Embeddings'])
        
        # Get the receiving candidates embeddings
        receivingCandidatesDF = receivingInventory.loc[receivingCandidates]
        receivingEmbeddings = np.vstack(receivingCandidatesDF['Embeddings'].values)
        
        # Calculate cosine distances
        distances = cosine_distances([sendingEmbedding], receivingEmbeddings)[0]
        
        # Store results
        results[sendingIdx] = (receivingCandidates, distances)
        
        # Log the best match
        bestMatchIdx = np.argmin(distances)
        bestMatchItem = receivingCandidatesDF.iloc[bestMatchIdx]['Item']
        bestDistance = distances[bestMatchIdx]
        
        logging.info(f'"{sendingItem["Item"]}" -> best embedding match: "{bestMatchItem}" (distance: {bestDistance:.3f})')
    
    return results


def assignOptimalMatches(embeddingDistances: dict, sendingInventory: pd.DataFrame, receivingInventory: pd.DataFrame) -> dict:
    """
    Assign optimal matches between sending and receiving items, handling conflicts by choosing the closest match.
    
    Args:
        embeddingDistances (dict): Dictionary from calculateEmbeddingDistancesForCandidates
        sendingInventory (pd.DataFrame): Sending items DataFrame
        receivingInventory (pd.DataFrame): Receiving items DataFrame
    
    Returns:
        dict: Dictionary mapping sending item indices to their best matched receiving item index
    """
    # Create a reverse mapping: receiving_item -> list of (sending_item, distance) tuples
    receivingToSending = {}
    
    for sendingIdx, (receivingIndices, distances) in embeddingDistances.items():
        for i, receivingIdx in enumerate(receivingIndices):
            distance = distances[i]
            if receivingIdx not in receivingToSending:
                receivingToSending[receivingIdx] = []
            receivingToSending[receivingIdx].append((sendingIdx, distance))
    
    # For each receiving item, find the closest sending item
    finalMatches = {}
    receivingItemAssignments = {}  # Track which receiving items are already assigned
    
    for receivingIdx, candidates in receivingToSending.items():
        # Sort by distance (closest first)
        candidates.sort(key=lambda x: x[1])
        
        # Find the first available sending item (not already assigned to a closer receiving item)
        for sendingIdx, distance in candidates:
            if sendingIdx not in finalMatches:
                finalMatches[sendingIdx] = receivingIdx
                receivingItemAssignments[receivingIdx] = sendingIdx
                break
    
    # Log the results
    for sendingIdx, receivingIdx in finalMatches.items():
        sendingItem = sendingInventory.loc[sendingIdx]['Item']
        receivingItem = receivingInventory.loc[receivingIdx]['Item']
        
        # Find the distance for this specific pair
        receivingCandidates, distances = embeddingDistances[sendingIdx]
        candidateIdx = receivingCandidates.index(receivingIdx)
        distance = distances[candidateIdx]
        
        logging.info(f'MATCH: "{sendingItem}" -> "{receivingItem}" (distance: {distance:.3f})')
    
    logging.info(f'Total matches found: {len(finalMatches)}')
    
    return finalMatches


if __name__ == "__main__":
    # Loading data
    filePattern = 'data/store_1_*.csv'
    receivingStoreInventory = pd.concat([pd.read_csv(f) for f in glob.glob(filePattern)], ignore_index=True)
    receivingStoreInventory['Qty.'] = pd.to_numeric(receivingStoreInventory['Qty.'].fillna(0), errors='coerce')
    receivingStoreInventory['Price'] = pd.to_numeric(receivingStoreInventory['Price'].fillna(0), errors='coerce')

    sendingStoreInventory = pd.read_csv('data/store_2.csv')
    sendingStoreInventory['Qty.'] = pd.to_numeric(sendingStoreInventory['Qty.'].fillna(0), errors='coerce')
    sendingStoreInventory = sendingStoreInventory.loc[sendingStoreInventory['Qty.'] > 0]

    # Cleaning data
    numberOfDuplicatesPrices, differentPrices = checkDuplicatesColumnDifferences(receivingStoreInventory, 'Price')
    numberOfDuplicatesUPC, differentUPC = checkDuplicatesColumnDifferences(receivingStoreInventory, 'UPC')
    receivingStoreInventory = mergeDuplicates(receivingStoreInventory, {'Qty.': 'sum', 'Price': 'mean', 'UPC': 'first'})
    receivingStoreInventory.to_csv('data/receivingStoreInventory.csv', index=False)

    sendingStoreInventory = mergeDuplicates(sendingStoreInventory, {'Qty.': 'sum'})
    sendingStoreInventory.to_csv('data/sendingStoreInventory.csv', index=False)
    
    # Adding embeddings
    receivingStoreInventory = addEmbeddingsToInventory(receivingStoreInventory)
    receivingStoreInventory.to_csv('data/receivingStoreInventory.csv', index=False)
    
    sendingStoreInventory = addEmbeddingsToInventory(sendingStoreInventory)
    sendingStoreInventory.to_csv('data/sendingStoreInventory.csv', index=False)
    
    # Get TF-IDF candidates to reduce calculation time
    matchingSendingInventoryCandidates = getTopCandidatesTFIDF(sendingStoreInventory, receivingStoreInventory, 50)
    
    # Getting best matches based on embedding distances
    embeddingDistances = calculateEmbeddingDistancesForCandidates(
        sendingStoreInventory, 
        receivingStoreInventory, 
        matchingSendingInventoryCandidates
    )
    
    optimalMatches = assignOptimalMatches(embeddingDistances, sendingStoreInventory, receivingStoreInventory)

    # Updating quantity of receiving inventory
    for sendingIdx, receivingIdx in optimalMatches.items():
        qtyToAdd = sendingStoreInventory.loc[sendingIdx, 'Qty.']
        receivingStoreInventory.loc[receivingIdx, 'Qty.'] += qtyToAdd

    receivingStoreInventory.to_csv('data/receivingStoreInventoryUpdated.csv', index=False)