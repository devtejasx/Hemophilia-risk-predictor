import React, { useState } from 'react'
import { Navbar } from '@/components/Navbar'
import { ChatBox } from '@/components/ChatBox'
import { chatAPI } from '@/services/api-client'
import { useAppStore } from '@/store/appStore'
import { MessageCircle } from 'lucide-react'
import clsx from 'clsx'

type ChatMode = 'clinical' | 'general' | 'treatment'

interface ConversationTopic {
  id: ChatMode
  label: string
  description: string
  emoji: string
}

export const Chatbot: React.FC = () => {
  const { chatMessages, addChatMessage, selectedChatMode, setChatMode, clearChatHistory } =
    useAppStore()
  const [loading, setLoading] = useState(false)

  const topics: ConversationTopic[] = [
    {
      id: 'clinical',
      label: 'Clinical Questions',
      description: 'Ask about hemophilia management and clinical care',
      emoji: '🏥',
    },
    {
      id: 'general',
      label: 'General Information',
      description: 'Learn about hemophilia, treatment options, and lifestyle',
      emoji: '📚',
    },
    {
      id: 'treatment',
      label: 'Treatment Planning',
      description: 'Discuss treatment strategies and medication options',
      emoji: '💊',
    },
  ]

  const handleSendMessage = async (message: string) => {
    try {
      setLoading(true)

      // Add user message
      addChatMessage('user', message)

      // Get AI response
      const response = await chatAPI.sendMessage(message, {
        mode: selectedChatMode,
        timestamp: new Date().toISOString(),
      })

      // Add assistant message
      addChatMessage('assistant', response.content || 'I could not generate a response.')
    } catch (error) {
      console.error('Chat error:', error)
      addChatMessage('assistant', 'Sorry, I encountered an error. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      <Navbar title="Clinical AI Chatbot" />

      <div className="p-6 max-w-6xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6 h-[600px]">
          {/* Sidebar - Topic Selection */}
          <div className="lg:col-span-1 space-y-4">
            <div>
              <h2 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
                Conversation Topics
              </h2>
              <div className="space-y-2">
                {topics.map((topic) => (
                  <button
                    key={topic.id}
                    onClick={() => {
                      setChatMode(topic.id)
                      clearChatHistory()
                    }}
                    className={clsx(
                      'w-full text-left p-4 rounded-lg transition border-2',
                      selectedChatMode === topic.id
                        ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                        : 'border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-purple-200'
                    )}
                  >
                    <div className="text-2xl mb-2">{topic.emoji}</div>
                    <h3 className="font-semibold text-slate-900 dark:text-white">
                      {topic.label}
                    </h3>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                      {topic.description}
                    </p>
                  </button>
                ))}
              </div>
            </div>

            {/* Quick Questions */}
            <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border border-slate-200 dark:border-slate-800">
              <h3 className="font-semibold text-slate-900 dark:text-white mb-3 text-sm">
                Quick Questions
              </h3>
              <div className="space-y-2">
                <button className="text-left text-xs text-purple-600 dark:text-purple-400 hover:underline w-full">
                  What is inhibitor development?
                </button>
                <button className="text-left text-xs text-purple-600 dark:text-purple-400 hover:underline w-full">
                  How often should I monitor?
                </button>
                <button className="text-left text-xs text-purple-600 dark:text-purple-400 hover:underline w-full">
                  What are treatment options?
                </button>
              </div>
            </div>
          </div>

          {/* Chat Area */}
          <div className="lg:col-span-3">
            <ChatBox
              messages={chatMessages}
              onSendMessage={handleSendMessage}
              loading={loading}
              placeholder={
                selectedChatMode === 'clinical'
                  ? 'Ask clinical questions...'
                  : selectedChatMode === 'treatment'
                    ? 'Discuss treatment options...'
                    : 'Ask general questions...'
              }
            />
          </div>
        </div>

        {/* Educational Content */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
            <h3 className="font-bold text-blue-900 dark:text-blue-200 mb-2">
              💡 Inhibitor Development
            </h3>
            <p className="text-sm text-blue-800 dark:text-blue-300">
              Factor inhibitors are antibodies that neutralize treatment. The chatbot can help explain
              risk factors and management strategies.
            </p>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
            <h3 className="font-bold text-green-900 dark:text-green-200 mb-2">
              🎯 Personalized Recommendations
            </h3>
            <p className="text-sm text-green-800 dark:text-green-300">
              Ask questions about your specific patient case and get evidence-based recommendations
              based on current clinical guidelines.
            </p>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-6">
            <h3 className="font-bold text-purple-900 dark:text-purple-200 mb-2">
              📖 Continuous Learning
            </h3>
            <p className="text-sm text-purple-800 dark:text-purple-300">
              Use the chatbot as an educational tool to continuously update your knowledge about
              hemophilia management.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Chatbot
